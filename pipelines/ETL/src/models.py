from typing import Dict, List, Optional

from pydantic import BaseModel


class Author(BaseModel):
    id: int
    name: str
    profile_url: str


class MediaMetadata(BaseModel):
    id: int
    url: str
    imgsize: List[int]
    image_url: str
    description: Optional[str]
    author: Author

    @classmethod
    def from_dict(cls, item: Dict) -> "MediaMetadata":
        return cls(
            id=item["id"],
            url=item["url"],
            imgsize=[item["width"], item["height"]],
            image_url=item["src"]["original"],
            description=item.get("description"),
            author=Author(
                id=item["photographer_id"],
                name=item["photographer"],
                profile_url=item["photographer_url"],
            ),
        )

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "url": self.url,
            "imgsize": self.imgsize,
            "image_url": self.image_url,
            "description": self.description,
            "author": {
                "id": self.author.id,
                "name": self.author.name,
                "profile_url": self.author.profile_url,
            },
        }
