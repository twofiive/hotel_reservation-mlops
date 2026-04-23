import os
import pandas as pd
import requests
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("DataIngestion initialisé")

    # ─────────────────────────────────────────
    # SOURCE 1 : Google Cloud Storage
    # ─────────────────────────────────────────
    def download_from_gcs(self):
        """
        Source 1 — Télécharge le dataset principal depuis GCS.
        Contient les réservations hôtelières brutes.
        """
        try:
            logger.info(f"[Source 1 - GCS] Téléchargement depuis gs://{self.bucket_name}/{self.file_name}")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"[Source 1 - GCS] Fichier téléchargé : {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"[Source 1 - GCS] Erreur : {e}")
            raise CustomException("Échec téléchargement GCS", e)

    # ─────────────────────────────────────────
    # SOURCE 2 : API REST Open-Meteo (météo historique)
    # ─────────────────────────────────────────
    def fetch_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Source 2 — API Open-Meteo (gratuite, sans clé API).
        Enrichit chaque réservation avec la météo à la date d'arrivée.
        Hôtel situé à Paris (lat=48.85, lon=2.35).
        Variables récupérées : température max, précipitations, vitesse vent.
        """
        logger.info("[Source 2 - API Open-Meteo] Début enrichissement météo")

        # Construire les dates uniques à interroger
        df["arrival_date_str"] = (
            df["arrival_year"].astype(str) + "-" +
            df["arrival_month"].astype(str).str.zfill(2) + "-" +
            df["arrival_date"].astype(str).str.zfill(2)
        )

        dates_uniques = df["arrival_date_str"].dropna().unique()
        date_min = dates_uniques.min()
        date_max = dates_uniques.max()

        logger.info(f"[Source 2 - API Open-Meteo] Période : {date_min} → {date_max}")

        try:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude=48.85&longitude=2.35"
                f"&start_date={date_min}&end_date={date_max}"
                f"&daily=temperature_2m_max,precipitation_sum,windspeed_10m_max"
                f"&timezone=Europe%2FParis"
            )
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Construire un DataFrame météo
            df_weather = pd.DataFrame({
                "arrival_date_str": data["daily"]["time"],
                "temperature_max": data["daily"]["temperature_2m_max"],
                "precipitation": data["daily"]["precipitation_sum"],
                "windspeed_max": data["daily"]["windspeed_10m_max"],
            })

            # Fusionner avec le dataset principal
            df = df.merge(df_weather, on="arrival_date_str", how="left")
            logger.info(f"[Source 2 - API Open-Meteo] {len(df_weather)} jours enrichis")

        except Exception as e:
            logger.warning(f"[Source 2 - API Open-Meteo] Erreur : {e} — colonnes météo vides")
            df["temperature_max"] = None
            df["precipitation"] = None
            df["windspeed_max"] = None

        df.drop(columns=["arrival_date_str"], inplace=True, errors="ignore")
        return df

    # ─────────────────────────────────────────
    # SOURCE 3 : OpenData — Tourisme INSEE
    # ─────────────────────────────────────────
    def fetch_tourism_opendata(self) -> pd.DataFrame:
        """
        Source 3 — API INSEE BDM (Banque de Données Macroéconomiques).
        Récupère les nuitées mensuelles dans les hôtels français.
        Série : 001694056 = Nuitées totales hôtellerie France métropolitaine
        Permet d'enrichir les réservations avec le contexte touristique mensuel.
        """
        logger.info("[Source 3 - API INSEE] Téléchargement nuitées hôtelières mensuelles")

        try:
            # API INSEE BDM - données mensuelles nuitées hôtellerie
            url = (
                "https://api.insee.fr/series/BDM/V1/data/SERIES_BDM/001694056"
                "?startPeriod=2017-01&endPeriod=2018-12"
            )
            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            observations = (
                data.get("GenericData", {})
                .get("DataSet", {})
                .get("Series", {})
                .get("Obs", [])
            )

            records = []
            for obs in observations:
                period = obs.get("ObsDimension", {}).get("value", "")
                value = obs.get("ObsValue", {}).get("value", None)
                if period:
                    year, month = period.split("-")
                    records.append({
                        "arrival_year": int(year),
                        "arrival_month": int(month),
                        "nuitees_hotels_france": float(value) if value else None
                    })

            df_tourism = pd.DataFrame(records)
            logger.info(f"[Source 3 - API INSEE] {len(df_tourism)} mois récupérés")
            
            os.makedirs(OPENDATA_DIR, exist_ok=True)
            df_tourism.to_csv(OPENDATA_FILE_PATH, index=False)
            
            return df_tourism

        except Exception as e:
            logger.warning(f"[Source 3 - API INSEE] Erreur : {e} — fallback CSV local")
            return self._load_tourism_fallback()

    def _load_tourism_fallback(self) -> pd.DataFrame:
        """
        Fallback : données nuitées hôtelières codées en dur pour 2017-2018.
        Source : INSEE enquête fréquentation hôtelière (données publiques).
        Utilisé si l'API INSEE est indisponible.
        """
        logger.info("[Source 3 - Fallback] Utilisation données nuitées statiques INSEE")
        
        # Nuitées mensuelles hôtellerie France (millions) 2017-2018 - Source INSEE
        data = {
            "arrival_year":  [2017]*12 + [2018]*12,
            "arrival_month": list(range(1, 13)) * 2,
            "nuitees_hotels_france": [
                # 2017 : Jan à Déc
                28.5, 26.1, 32.4, 35.2, 38.7, 42.1,
                52.3, 54.8, 41.2, 37.6, 30.1, 35.8,
                # 2018 : Jan à Déc
                29.1, 27.3, 33.8, 36.4, 39.9, 43.5,
                53.7, 55.2, 42.8, 38.9, 31.4, 37.2
            ]
        }
        df = pd.DataFrame(data)
        
        os.makedirs(OPENDATA_DIR, exist_ok=True)
        df.to_csv(OPENDATA_FILE_PATH, index=False)
        logger.info(f"[Source 3 - Fallback] {len(df)} lignes sauvegardées")
        return df
    
    # ─────────────────────────────────────────
    # PIPELINE COMPLET
    # ─────────────────────────────────────────
    def split_data(self, df: pd.DataFrame):
        """Découpe le dataset enrichi en train/test et sauvegarde."""
        try:
            train_data, test_data = train_test_split(
                df,
                test_size=1 - self.train_test_ratio,
                random_state=42
            )
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Train : {len(train_data)} lignes | Test : {len(test_data)} lignes")
        except Exception as e:
            logger.error(f"Erreur split : {e}")
            raise CustomException("Échec split data", e)

    def run(self):
        try:
            logger.info("=== Démarrage pipeline ingestion ===")

            # Source 1 — GCS
            self.download_from_gcs()
            df = pd.read_csv(RAW_FILE_PATH)
            logger.info(f"Dataset principal : {df.shape}")

            # Source 2 — API météo
            df = self.fetch_weather_data(df)
            logger.info(f"Après enrichissement météo : {df.shape}")

            # Source 3 — Nuitées hôtelières INSEE
            df_tourism = self.fetch_tourism_opendata()
            if not df_tourism.empty:
                df = df.merge(
                    df_tourism,
                    on=["arrival_year", "arrival_month"],
                    how="left"
                )
                logger.info(f"[Source 3] Après enrichissement tourisme : {df.shape}")

            # Sauvegarde dataset enrichi + split
            df.to_csv(RAW_FILE_PATH, index=False)
            self.split_data(df)

            logger.info("=== Pipeline ingestion terminé ===")
            logger.info(f"Colonnes finales : {list(df.columns)}")

        except CustomException as ce:
            logger.error(f"Erreur pipeline : {ce}")
        finally:
            logger.info("DataIngestion terminé")


if __name__ == "__main__":
    ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    ingestion.run()