# License

COCO annotations are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode) (see COCO [terms of use](https://cocodataset.org/#termsofuse)).

In order to make this tutorial faster to run and not require a large download, we've provided a modified annotations JSON file from the validation split of the COCO 2017 Object Detection Task containing only the first 4 images (and will dynamically download only those images using the "coco_url").

Here's the code used to create the subset of the annotations file:

```python
def annotation_subset(annotation_file: Path, num_images: int) -> dict[str, Any]:
    # keys: info, licenses, images, annotations, categories
    d = json.load(open(annotation_file, "r"))

    # get set of image IDs to keep
    assert num_images <= len(d["images"])
    image_ids = {img["id"] for img in d["images"][:num_images]}

    # only keep images with those IDs
    d["images"] = [img for img in d["images"] if img["id"] in image_ids]

    # only keep annotations for images with those IDs
    d["annotations"] = [ann for ann in d["annotations"] if ann["image_id"] in image_ids]

    return d

# original annotations file
annotation_file = Path("instances_val2017.json")

# subset
d_sub = annotation_subset(annotation_file, 4)
subset_file = Path("instances_val2017_first4.json")
with open(subset_file, "w") as outfile:
    json.dump(d_sub, outfile)
```

