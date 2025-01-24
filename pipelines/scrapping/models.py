import datetime
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, HttpUrl
from pydantic_settings import BaseSettings


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
    avg_color: str = Field(default=None) # hex color red color
    height: int = Field(ge=720)
    id: int = Field(default=None)
    liked: bool = Field(default=False)
    photographer: str = Field(default=None)
    photographer_id: int = Field(required=True)
    photographer_url: str = Field(default=None)
    src: Dict[str, str] = Field(default=None)
    url: str = Field(required=True)
    width: int = Field(ge=1280)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "PexelsItem":
        try:
            # return cls(**json_data)
            return PexelsItem(**json_data)
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
    created_at: str = Field(default=None)
    current_user_collections: List[str] = Field(default=[])
    description: Union[str, None] = Field(default=None)
    height: int = Field(ge=720)
    id: str = Field(default="")
    liked_by_user: bool = Field(default=False)
    likes: int = Field(default=0)
    links: Dict[str, str] = Field(default=None)
    promoted_at: Optional[str] = Field(default=None)
    slug: str = Field(default=None)
    sponsorship: Optional[str] = Field(default=None)
    topic_submissions: Dict[str, Any] = Field(default=None)
    updated_at: str = Field(default=None)
    urls: Dict[str, str] = Field(default=None)
    user: Dict[str, Any] = Field(default=None)
    width: int = Field(ge=1280)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "UnsplashItem":
        try:
            return cls(**json_data)
        except ValueError as e:
            raise ValueError(f"Error creating UnsplashItem from JSON: {e}")

class CommonMediaDocument(BaseModel):
    media_id: str
    weburl: str
    imgsize: List[int]
    image_url: str
    description: Optional[str]
    author: Author
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    from_page_num: int = Field(default=1)
    subject: str = Field(required=True, description="The search subject the image belongs to.")
 
    @classmethod
    def from_pexels(cls, item: PexelsItem, subject: str,  page_number: int) -> "CommonMediaDocument":
        return cls(
            media_id=str(item.id),
            weburl=item.url,
            imgsize=[item.width, item.height],
            image_url=item.src["original"],
            description=item.alt,
            author=Author(
                id=str(item.photographer_id),
                username=item.photographer,
            ),
            from_page_num=page_number,
            subject=subject,
        )
    
    @classmethod
    def from_unsplash(cls, item: UnsplashItem, subject: str, page_number: int) -> "CommonMediaDocument":
        return cls(
            media_id=str(item.id),
            weburl=item.links["html"],
            imgsize=[item.width, item.height],
            image_url=item.urls["full"],
            description=item.alt_description,
            author=Author(
                id=str(item.user["id"]),
                username=item.user["username"]
            ),
            from_page_num=page_number,
            subject=subject,
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
                "username": self.author.username,
            },
        }

# MONGO stores entries as BSON, which requires a specific type for the primary key.
# This is a `str` that will be converted to a `bson.ObjectId` when sent to MongoDB.
PyObjectId = Annotated[str, BeforeValidator(str)]

class RawMediaMongoDocument(BaseModel):
    """
    Container for a MediaDocument in MongoDB collection.
    """

    # The primary key for the RawMediaMongoDocument, stored as a `str` on the instance.
    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the previous Pydantic models we have ID as string
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    weburl: HttpUrl = Field(..., description="The URL of the image.")
    imgsize: List[int] = Field(..., description="The [w,h] size of the image.")
    image_url: str = Field(..., description="Full URL to image.")
    description: Optional[str] = Field(None, description="Image Description/")
    author: Author = Field(..., description="Author name and username")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now, description="TS when document was created")
    from_page_num: Optional[int] = Field(None, description="The page number where the image was found.")
    subject: str = Field(..., description="The search subject the image belongs to.")
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "weburl": "https://unsplash.com/photos/73yZU5D5aFQ",
                "imgsize": [1920, 1080],
                "image_url": "https://images.unsplash.com/photo-1566557087503-b839ce6e5aa0?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2Nzk0Nzl8MHwxfHNlYXJjaHw4fHxib3R0bGVzfGVufDB8fHx8MTczNzcyMTk1Nnww&ixlib=rb-4.0.3&q=85",
                "description": "A photo of a glass bottle on a table.",
                "author": {
                    "id": "Jane Doe",
                    "username": "jdoe",
                },
                "created_at": "2021-09-01T12:00:00",
                "from_page_num": 1,
                "subject": "glass bottle",
            }
        },
    )
    
    @classmethod
    def from_commondoc(cls, doc: CommonMediaDocument) -> "RawMediaMongoDocument":
        return cls(
            weburl=doc.weburl,
            imgsize=doc.imgsize,
            image_url=doc.image_url,
            description=doc.description,
            author=doc.author,
            created_at=doc.created_at,
            from_page_num=doc.from_page_num,
            subject=doc.subject,
        )