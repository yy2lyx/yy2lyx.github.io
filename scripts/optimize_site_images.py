from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image, ImageOps


@dataclass
class SiteImageOptimizer:
    root: Path
    thumbnail_size: tuple = (640, 360)
    hero_width: int = 1920
    webp_quality: int = 78

    def __call__(self):
        return self.run()

    def optimize_thumbnail(self, path):
        output_path = path.with_name(f'{path.stem}-thumb.webp')
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image = ImageOps.fit(image, self.thumbnail_size, method=Image.Resampling.LANCZOS)
            image.save(output_path, 'WEBP', quality=self.webp_quality, method=6, exif=b'')

    def optimize_hero(self, path):
        output_path = path.with_name(f'{path.stem}-1920.webp')
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            height = round(image.height * self.hero_width / image.width)
            image = image.resize((self.hero_width, height), Image.Resampling.LANCZOS)
            image.save(output_path, 'WEBP', quality=self.webp_quality, method=6, exif=b'')

    def run(self):
        cover_pattern = re.compile(r'^cover:\s*["\']?([^"\'\n]+)', re.MULTILINE)
        covers = set()
        for post_path in (self.root / '_posts').glob('*.md'):
            covers.update(cover_pattern.findall(post_path.read_text(encoding='utf-8')))

        for cover in sorted(covers):
            self.optimize_thumbnail(self.root / cover.strip())

        self.optimize_thumbnail(self.root / 'assets' / 'images' / 'blog-cover.jpg')

        for name in ('home-network.jpg', 'home-hand.jpg', 'home-code.jpg', 'home-toolkit.jpg'):
            self.optimize_hero(self.root / 'assets' / 'images' / name)


if __name__ == '__main__':
    SiteImageOptimizer(Path(__file__).resolve().parent.parent)()
