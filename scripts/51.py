import fiftyone as fo
import fiftyone.zoo as foz
import json
import pickle as pkl
from pathlib import Path
import torch
from torchvision.ops.boxes import box_convert, box_iou
from tqdm import tqdm

"""
{
    "colorPool": [
        "#ee0000",
        "#ee6600",
        "#993300",
        "#996633",
        "#999900",
        "#009900",
        "#003300",
        "#009999",
        "#000099",
        "#0066ff",
        "#6600ff",
        "#cc33cc",
        "#777799"
    ],
    "colorBy": "value",
    "opacity": 0.7,
    "multicolorKeypoints": false,
    "showSkeletons": true,
    "fields": [
        {
            "path": "detections",
            "fieldColor": null,
            "colorByAttribute": "label",
            "valueColors": [
                {
                    "color": "#0505D6",
                    "value": "car"
                },
                {
                    "color": "#1AED1A",
                    "value": "truck"
                },
                {
                    "color": "#E10A0A",
                    "value": "bus"
                },
                {
                    "color": "#20F4F4",
                    "value": "van"
                },
                {
                    "color": "#E612E6",
                    "value": "freight_car"
                }
            ],
            "maskTargetsColors": []
        },
        {
            "path": "prediction",
            "fieldColor": null,
            "colorByAttribute": "label",
            "valueColors": [
                {
                    "color": "#0900ee",
                    "value": "car"
                },
                {
                    "color": "#7edd6a",
                    "value": "truck"
                },
                {
                    "color": "#990020",
                    "value": "bus"
                },
                {
                    "color": "#1ce3e1",
                    "value": "van"
                },
                {
                    "color": "#ed1cf2",
                    "value": "freight_car"
                }
            ],
            "maskTargetsColors": []
        }
    ],
    "labelTags": null,
    "defaultMaskTargetsColors": null,
    "colorscales": [],
    "defaultColorscale": {
        "name": "viridis",
        "list": null
    }
}
"""


# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example
label_name = ('car', 'truck', 'bus', 'van', 'freight_car')
workspace = '/home/renzhen/userdata/repo/gaiic2024'
ds_name = "gaiic_test"

if __name__ == "__main__":
    if ds_name in fo.list_datasets():
        fo.delete_dataset(ds_name)
    dataset = fo.Dataset.from_dir(
        data_path=f'{workspace}/data/track1-A/test/rgb',
        labels_path=f'{workspace}/data/track1-A/annotations/test.json',
        dataset_type=fo.types.COCODetectionDataset,
        name=ds_name,
        #max_samples=1000,
        persistent=True
    )
    # dataset = fo.load_dataset("gaiic_train")

    pred = pkl.load(open(f"{workspace}/work_dirs/codetr_all_in_one/_20240520_153439/epoch_10_submit.pkl", 'rb'))
    for x in tqdm(pred, desc='add predictions'):
        inst = x['pred_instances']
        # sample = dataset[str( Path(workspace) / x['img_path'] ).replace('rgb', 'tir')]
        sample = dataset[str( Path(workspace) / x['img_path'] )]
        h, w = x['ori_shape']
        wh = torch.tensor([w, h, w, h], dtype=torch.float)
        
        boxes = box_convert(inst['bboxes'], 'xyxy', 'xywh') / wh

        sample['prediction'] = fo.Detections(detections=[
            fo.Detection(
                label=label_name[inst['labels'][i].item()],
                bounding_box=boxes[i].tolist(),
                confidence=inst['scores'][i].item(),
            ) for i in range(len(boxes))
        ])
        sample.save()

    session = fo.launch_app(dataset, remote=True)
    session.wait(-1)
