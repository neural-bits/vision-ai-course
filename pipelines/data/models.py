from typing import Dict, List, Optional, Any, Union

import datetime
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from loguru import logger

# Configuration
class Config(BaseSettings):
    unsplash_api_key: str
    pexels_api_key: str
    mongo_uri: str

    class Config:
        env_file = ".env"

class Author(BaseModel):
    id: str
    username: str


class PexelsItem(BaseModel):
    alt: str = Field(default=None)
    avg_color: str = Field(default=None)
    height: int = Field(ge=1080)
    id: int = Field(default=None)
    liked: bool = Field(default=False)
    photographer: str = Field(default=None)
    photographer_id: int = Field(required=True)
    photographer_url: str = Field(default=None)
    src: Dict[str, str] = Field(default=None)
    url: str = Field(required=True)
    width: int = Field(ge=1920)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "PexelsItem":
        try:
            return cls(**json_data)
        except ValueError as e:
            print(f"Error creating PexelsItem from JSON: {e}")
            return None 
        
    
class UnsplashItem(BaseModel):
    alt_description: str = Field(default="")
    alternative_slugs: Dict[str, str] = Field(default=None)
    asset_type: str = Field(default="photo")
    blur_hash: str = Field(default=None)
    breadcrumbs: List[str] = Field(default=[])
    color: str  = Field(default=None)
    created_at: str = Field(default_factory=datetime.datetime.now)
    current_user_collections: List[str] = Field(default=[])
    description: Union[str, None] = Field(default=None)
    height: int = Field(ge=1080)
    id: str = Field(default="")
    liked_by_user: bool = Field(default=False)
    likes: int = Field(default=0)
    links: Dict[str, str] = Field(default=None)
    promoted_at: Optional[str] = Field(default=None)
    slug: str = Field(default=None)
    sponsorship: Optional[str] = Field(default=None)
    topic_submissions: Dict[str, Any] = Field(default=None)
    updated_at: str = Field(default_factory=datetime.datetime.now)
    urls: Dict[str, str] = Field(default=None)
    user: Dict[str, Any] = Field(default=None)
    width: int = Field(ge=1920)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "UnsplashItem":
        try:
            return cls(**json_data)
        except ValueError as e:
            print(f"Error creating UnsplashItem from JSON: {e}")
            return None 

class CommonMediaDocument(BaseModel):
    id: str
    url: str
    imgsize: List[int]
    image_url: str
    description: Optional[str]
    author: Author

    @classmethod
    def from_pexels(cls, item: PexelsItem) -> "CommonMediaDocument":
        return cls(
            id=str(item.id),
            url=item.url,
            imgsize=[item.width, item.height],
            image_url=item.src["original"],
            description=item.alt,
            author=Author(
                id=str(item.photographer_id),
                username=item.photographer,
            ),
        )
    
    @classmethod
    def from_unsplash(cls, item: UnsplashItem) -> "CommonMediaDocument":
        return cls(
            id=str(item.id),
            url=item.links["html"],
            imgsize=[item.width, item.height],
            image_url=item.urls["full"],
            description=item.alt_description,
            author=Author(
                id=str(item.user["id"]),
                username=item.user["username"]
            ),
        )

    def to_json(self) -> Dict:
        return {
            "id": self.id,
            "url": self.url,
            "imgsize": self.imgsize,
            "image_url": self.image_url,
            "description": self.description,
            "author": {
                "id": self.author.id,
                "username": self.author.name,
            },
        }
