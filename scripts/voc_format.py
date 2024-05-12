from xml.dom.minidom import Document
from typing import Optional, List
import xml.etree.ElementTree as ET
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm


def extract_tag(root, tag, default=None):
    if root is not None:
        x = root.find(tag)
        if x is not None:
            return x.text
    return default

def new_text_element(doc, tag, text):
    e = doc.createElement(tag)
    e.appendChild(doc.createTextNode(str(text)))
    return e


class XmlAnno:
    """
    替代 CreateAnno
    """
    class Size:
        def __init__(self, *args, **kwargs) -> None:
            if args:
                self.width, self.height, self.depth = args
            else:
                self.width = None
                self.height = None
                self.depth = None
            self.__dict__.update(kwargs)

        def to_dict(self):
            return {
                'width': self.width,
                'height': self.height,
                'depth': self.depth
            }

        def __repr__(self):
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    class AnnoBox:
        def __init__(self, **kwargs) -> None:
            self.name: str = ''
            self.pose = "Unspecified"
            self.truncated = '0'
            self.difficult = '0'
            self.extra :str = ''
            self.bndbox = []        # [xmin, ymin, xmax, ymax]
            self.__dict__.update(kwargs)
            
        def get_property(self, idx):
            # TODO: 目前假设只有红绿灯会调用该方法。不一定兼容其他类型(或extra格式)
            if self.is_ignore():
                return '?'
            if len(self.extra) > idx:
                return self.extra[idx]
            return '?'
        
        def is_ignore(self) -> bool:
            return 'ignore' in self.name.lower() or 'none' in self.name.lower()
            
        def min_edge(self):
            xmin, ymin, xmax, ymax = self.bndbox
            return min(xmax - xmin, ymax - ymin)
        
        @classmethod
        def parse(cls, obj):
            box = XmlAnno.AnnoBox(
                name=extract_tag(obj, 'name'), 
                pose=extract_tag(obj, 'pose', 'Unspecified'), 
                truncated=extract_tag(obj, 'truncated', '0'),
                difficult=extract_tag(obj, 'difficult', '0'),
                extra=extract_tag(obj, 'extra')
            )
            xmlbox = obj.find('bndbox')
            robndbox = obj.find('robndbox')
            if xmlbox:
                box.bndbox = [
                    round(float(xmlbox.find('xmin').text)),
                    round(float(xmlbox.find('ymin').text)),
                    round(float(xmlbox.find('xmax').text)),
                    round(float(xmlbox.find('ymax').text))
                ]
            else:
                if robndbox:
                    box.bndbox = [
                        round(float(robndbox.find('cx').text) - float(robndbox.find('w').text) / 2), 
                        round(float(robndbox.find('cy').text) - float(robndbox.find('h').text) / 2),
                        round(float(robndbox.find('cx').text) + float(robndbox.find('w').text) / 2),
                        round(float(robndbox.find('cy').text) + float(robndbox.find('h').text) / 2)
                    ]
            if box.extra == '' or box.extra == "None":
                box.extra = ''
            if box.name and (xmlbox or robndbox):
                return box
            return None

        def to_dict(self):
            return {
                'name': self.name,
                'pose': self.pose,
                'truncated': self.truncated,
                'difficult': self.difficult,
                'extra': self.extra,
                'bndbox': self.bndbox
            }

        def __repr__(self):
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __init__(self, **kwargs) -> None:
        self.folder = None
        self.filename = None
        self.path = None
        self.source = None
        self.size :Optional[XmlAnno.Size] = None
        self.segmented = None
        self.objects :List[XmlAnno.AnnoBox] = []
        self._parse_from :Optional[str] = None
        self.__dict__.update(kwargs)

    def __repr__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_dict(self):
        return {
            'folder': self.folder,
            'filename': self.filename,
            'path': self.path,
            'source': self.source,
            'size': None if self.size is None else self.size.to_dict(),
            'segmented': self.segmented,
            'objects': [x.to_dict() for x in self.objects]
        }

    def save(self, xmlfile):
        doc = Document()
        anno = doc.createElement('annotation')
        if self.folder:
            anno.appendChild(new_text_element(doc, 'folder', self.folder))
        if self.filename:
            anno.appendChild(new_text_element(doc, 'filename', self.filename))
        if self.path:
            anno.appendChild(new_text_element(doc, 'path', self.path))
        if self.source:
            s = doc.createElement('source')
            s.appendChild(new_text_element(doc, 'database', self.source))
            anno.appendChild(s)
        if self.segmented is not None:
            anno.appendChild(new_text_element(doc, 'segmented', self.segmented))
        if self.size:
            size = doc.createElement('size')
            size.appendChild(new_text_element(doc, 'width', self.size.width))
            size.appendChild(new_text_element(doc, 'height', self.size.height))
            size.appendChild(new_text_element(doc, 'depth', self.size.depth))
            anno.appendChild(size)
        for obj in self.objects:
            x = doc.createElement('object')
            x.appendChild(new_text_element(doc, 'name', obj.name))
            x.appendChild(new_text_element(doc, 'pose', obj.pose))
            x.appendChild(new_text_element(doc, 'truncated', obj.truncated))
            x.appendChild(new_text_element(doc, 'difficult', obj.difficult))
            x.appendChild(new_text_element(doc, 'extra', obj.extra))

            bndbox = doc.createElement('bndbox')
            xmin, ymin, xmax, ymax = obj.bndbox
            bndbox.appendChild(new_text_element(doc, 'xmin', xmin if isinstance(xmin, str) else str(round(xmin))))
            bndbox.appendChild(new_text_element(doc, 'ymin', ymin if isinstance(ymin, str) else str(round(ymin))))
            bndbox.appendChild(new_text_element(doc, 'xmax', xmax if isinstance(xmax, str) else str(round(xmax))))
            bndbox.appendChild(new_text_element(doc, 'ymax', ymax if isinstance(ymax, str) else str(round(ymax))))

            x.appendChild(bndbox)
            anno.appendChild(x)
        doc.appendChild(anno)
        with open(xmlfile, "w") as f:
            doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')

    @classmethod
    def parse(cls, xmlfile):
        root = ET.parse(str(xmlfile)).getroot()
        size = root.find('size')
        w = int(extract_tag(size, 'width'))
        h = int(extract_tag(size, 'height'))
        d = int(extract_tag(size, 'depth'))
        res = XmlAnno(
                folder=extract_tag(root, 'folder', ''),
                filename=extract_tag(root, 'filename', ''),
                path=extract_tag(root, 'path', ''),
                source=extract_tag(root.find('source'), 'database', ''),
                segmented=extract_tag(root, 'segmented', '0'),
                size=XmlAnno.Size(width=w, height=h, depth=d),
                _parse_from=str(xmlfile)
        )
        for obj in root.iter('object'):
            t = XmlAnno.AnnoBox.parse(obj)
            if t:
                res.objects.append(t)
        return res


if __name__ == "__main__":
    coco = COCO('/home/renzhen/userdata/repo/gaiic2024/data/track1-A/annotations/train.json')
    save_dir = Path("/home/renzhen/userdata/repo/gaiic2024/data/track1-A/train/xml")
    save_dir.mkdir(parents=True, exist_ok=True)

    LABEL_NAMES = ('[PLACEHOLDER]', 'car', 'truck', 'bus', 'van', 'freight_car')
    for img_id in tqdm(coco.getImgIds()):
        ann = coco.loadAnns(coco.getAnnIds(img_id))
        img = coco.loadImgs(img_id)[0]
        f = Path(img['file_name'])
        voc = XmlAnno(
            filename=img['file_name'], 
            size=XmlAnno.Size(width=img['width'], height=img['height'], depth=3),
        )
        voc.objects = [
            XmlAnno.AnnoBox(name=LABEL_NAMES[x['category_id']], bndbox=[x['bbox'][0], x['bbox'][1], x['bbox'][0] + x['bbox'][2], x['bbox'][1] + x['bbox'][3]])
            for x in ann
        ]

        voc.save(save_dir / f"{f.stem}.xml")
