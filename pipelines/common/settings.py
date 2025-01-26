from pydantic_settings import BaseSettings, SettingsConfigDict


class APIScrappingSettings(BaseSettings):
    unsplash_api_key: str
    pexels_api_key: str

    api_image_subject: str
    api_image_last_page: int
    api_image_num_pages: int
    api_image_num_items_per_page: int


class MongoSettings(BaseSettings):
    mongo_uri: str
    mongo_db_name: str
    raw_data_collection: str
    clean_data_collection: str
    dataset_collection: str


class S3Settings(BaseSettings):
    aws_s3_datalake_bucket: str
    aws_s3_datawarehouse_bucket: str
    aws_access_key_id: str
    aws_secret_access_key: str


def get_mongo_settings():
    return MongoSettings(_env_file=".secrets/env-db.env")


def get_s3_settings():
    return S3Settings(_env_file=".secret/env-storage.env")


def get_api_scrapping_settings():
    return APIScrappingSettings(_env_file=".secrets/env-scrapping.env")
