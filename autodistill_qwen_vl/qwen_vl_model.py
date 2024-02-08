import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class QwenVL(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True, device_map=DEVICE
        )

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True
        ).eval()

        self.model = model
        self.tokenizer = tokenizer

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        query = self.tokenizer.from_list_format(
            [
                {
                    "image": input,
                },
                {"text": "Generate the caption in English with grounding and reference to the ontology: {}".format(self.ontology.prompts())},
            ]
        )
        
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        print(response)
        # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
        # get bboxes of each class
        xyxys = []
        confidences = []
        class_ids = []

        for prompt in self.ontology.prompts():
            if "<ref> {}</ref>".format(prompt) in response:
                box = response.split(prompt)[1].split("<box>")[1].split("</box>")[0]
                box = box.replace("(", "").replace(")", "")
                box = box.split(",")
                box = [int(i) for i in box]
                xyxys.append(box)
                confidences.append(1)
                class_ids.append(self.ontology.prompts()[prompt])

        return sv.Detections(
            xyxy=np.array(),
            confidence=np.array(),
            class_id=np.array(),
        )
