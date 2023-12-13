.. meta::
   :description: Tutorial on wrapping an object detection dataset.


================================
Wrap an Object Detection Dataset
================================

In this tutorial, we will walk through the steps to:

1. Use HuggingFace's ``datasets.load_dataset`` to load images from a folder
2. Add truth annotations to the dataset
3. Wrap the dataset to conform to ``MAITE`` protocols
4. Validate the dataset conforms to MAITE protocols


1. Load VisDrone with Huggingface API
=====================================

The VisDrone dataset is a large-scale benchmark contains annotated ground truth data for various drone-based computer vision tasks. 
Here we focus on the drone images for object detection task. You can find more information about data in <https://github.com/VisDrone/VisDrone-Dataset>.

To load the dataset we will utilize the "imagefolder" data loader provided
by `HuggingFace <https://huggingface.co/docs/datasets/loading>`_:

.. code-block:: pycon

    >>> from datasets import load_dataset
    >>> path = "<path to VisDrone dataset>"
    >>> vis_dataset = load_dataset("imagefolder", data_dir=path, split="test") 

The keyword arguments are to specify loading images from a folder, the location of the dataset, and to utilize the "test" split of the dataset.
Be sure to consult the documentation from the provider for more information on what parameters are available.

Here is what the dataset looks like:

.. code-block:: pycon

    >>> vis_dataset
    Dataset({
        features: ["image"],
        num_rows: 548
    })

Notice that the type of the image is a pillow `ImageFile`:

.. code-block:: pycon

    >>> vis_dataset[0]['image']
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080 at 0x7F8F4C0B3D90>


2. Customize a Function to Load Objects
=======================================

Here we provide a function to read the ``annotation`` folder to get the truth detection objects for
for the Visdrone dataset.

.. dropdown:: Create Annotation Objects

    .. code-block:: python
    
        import typing as t
        from pathlib import Path
        import torch as tr
        import maite.protocols as pr
        from torchvision.ops.boxes import box_convert

        def create_objects(
            dataset: pr.Dataset[pr.HasDataImage],
            label_map: t.Dict[int, int] | None = None,
            box_format: str = "xywh",
            annotation_folder: str = "annotations",
            **kwargs: t.Any
        ) -> t.Sequence[pr.HasDataObjects]:
            """
            Creates a list of object detections from an ImageFolderDict.
        
            Parameters
            ----------
            dataset : Dataset
                The dataset to add the objects to.
            label_map: t.Dict[int, int] | None = None
                A dictionary mapping the original labels to new labels.
            box_format : str
                The format of the bounding boxes. Defaults to "xywh".
            annotation_folder : str
                The name of the folder containing the annotations. Defaults to "annotations".
        
            Returns
            -------
            List[HasDataObjects]
            """
            objects = []
            for image in dataset["image"]:
                f = (
                    Path(image.filename).parent.parent
                    / annotation_folder
                    / f"{Path(image.filename).name.split('.')[0]}.txt"
                )
                with open(f, "r") as file:  # read annotation.txt
                    boxes, categories = [], []
                    for row in [x.split(",") for x in file.read().strip().splitlines()]:
                        if row[4] == "0":  # VisDrone 'ignored regions' class 0
                            continue

                        label = int(row[5]) - 1
                        if label_map is not None:
                            label = label_map[label]
                            categories.append(label)

                        box = tr.tensor(list(map(int, row[:4])))
                        if box_format != "xyxy":
                            box = box_convert(box, box_format, "xyxy")
                        boxes.append(box.cpu().tolist())

                    objects.append(pr.HasDataObjects(boxes=boxes, labels=categories))
            return objects

The original VisDrone dataset has 10 classes, since class zero is ``ignored_regions``, we reordered the rest of 9 classes as follows
to conform with the COCO label mappings:

.. code-block:: python

    id2label = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor",
    }

By definint the new label mapping, we can create the VisDrone object detections are that
conform to the COCO label mappings:

.. code-block:: pycon

    >>> visdrone_to_coco_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 7, 5: 7, 6: 3, 7: 3, 8: 5, 9 : 3}
    >>> vis_dataset = vis_dataset.add_column("objects", create_objects(vis_dataset, label_map=visdrone_to_coco_mapping))
    >>> vis_dataset
    Dataset({
        features: ['image', 'objects'],
        num_rows: 548
    })

Here is an example of accessing the first three images of the dataset:

.. code-block:: pycon

    >>> vis_dataset[:3]['image']
    [
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080>,
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080>,
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080>,
    ]

Here is an example output of the ``objects`` data:

.. code-block:: pycon

    >>> vis_dataset[0]['objects']
    {
        'boxes': [
            [871, 572, 925, 664],
            [948, 592, 1010, 684],
            [874, 705, 941, 815]],
            ...
        'labels': [
            3,
            3,
            3,
            ...
        ]
    }

3. Create a :class:`~maite.protocols.ObjectDetectionDataset`
====================================================================

To conform with :class:`~maite.protocols.ObjectDetectionDataset`, a dataset's
output must support :class:`~maite.protocols.SupportsObjectDetection`, a dictionary
requiring the following keys:

- `images`: a :class:`~maite.protocols.SupportsArray` type
- `objects`: a :class:`~maite.protocols.HasDataObjects` type

By construction the `objects` key is already supported by the dataset.  However, the `image` key is not.
We can update by defining a transform function that converts the image to 
a :class:`~maite.protocols.SupportsArray` type simply by converting the image to a numpy array:


.. code-block:: python

    import numpy as np
    
    def transform_pil2numpy(x):
        x.update(image=[np.asarray(i) for i in x['image']])
        return x


Now lets pull this all together into a dataset that conforms to 
the :class:`~maite.protocols.ObjectDetectionDataset` protocol:

.. code-block:: python

    class VisDronDataset:
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset

        @classmethod
        def from_hf_dataset(cls, path: str | Path, **kwargs):
            vis_dataset = load_dataset("imagefolder", path=path, **kwargs)
            vis_dataset = vis_dataset.add_column(
                "objects", 
                create_objects(vis_dataset, label_map=visdrone_to_coco_mapping)
            )
            vis_dataset.set_tr
            vis_dataset.set_transform(transform_pil2numpy)
            return cls(vis_dataset)
        
        def __len__(self) -> int:
            return len(self.hf_dataset)
            
        def __getitem__(self, index: int) -> pr.SupportsObjectDetection:
            data = self.hf_dataset[index]
            return pr.SupportsObjectDetection(
                image = data["image"],
                objects = data["objects"],
            )

4. Validate the Dataset
=======================

We can validate the dataset conforms to the :class:`~maite.protocols.ObjectDetectionDataset` protocol using 
both static type checking and runtime validation:

.. code-block:: python

    import maite.protocols as pr
    import typing as t

    VisDrone_dataset = VisDronDataset.from_hf_dataset(path, split="test")

    # type checking
    if t.TYPE_CHECKING:
        def f(dataset: pr.Dataset[pr.SupportsObjectDetection]):
            ...
            
        # passes
        f(VisDrone_dataset)

    # runtime validation
    assert isinstance(VisDrone_dataset, pr.Dataset)

    example_output = VisDrone_dataset[0]
    assert isinstance(example_output, dict)
    assert "image" in example_output
    assert "objects" in example_output

    assert isinstance(example_output["image"], pr.ArrayLike)
    assert example_output["image"].shape[-1] == 3
    assert example_output["image"].dtype == np.uint8
