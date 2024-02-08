<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Qwen-VL Module

This repository contains the code supporting the Qwen-VL base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Qwen-VL](https://qwenlm.github.io/blog/qwen-vl/), introduced in the paper [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966), is a multimodal vision model. Qwen-VL has visual grounding capabilities, which allows you to use the model for zero-shot object detection.

You can use Autodistill Qwen-VL to auto-label images for use in training a smaller, fine-tuned vision model.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Qwen-VL Autodistill documentation](https://autodistill.github.io/autodistill/base_models/qwen-vl/).

## Installation

To use Qwen-VL with Autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-qwen-vl
```

## Quickstart

```python
from autodistill_qwen_vl import QwenVL
from autodistill.utils import plot
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our QwenVL prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = QwenVL(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
results = base_model.predict("logistics.jpeg")

plot(
    image=cv2.imread("logistics.jpeg"),
    classes=base_model.ontology.classes(),
    detections=results
)

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```

## License

[add license information here]

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!