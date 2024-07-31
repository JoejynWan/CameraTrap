# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from .base_classifier import PlainResNetInference

__all__ = [
    "SINClassifier"
]


class SINClassifier(PlainResNetInference):
    """
    Singapore Animal Classifier that inherits from PlainResNetInference.
    This classifier is specialized for recognizing 35 different groups of animals in Singapore's 
    forests, split at the family/order level.
    """
    
    # Image size for SINClassifier
    IMAGE_SIZE = 224
    
    # Class names for prediction
    CLASS_NAMES = {
        0: 'Bats (Chiroptera)',
        1: 'BitternsHerons (Pelecaniformes)',
        2: 'Boars (Suidae)',
        3: 'Cats (Felidae)',
        4: 'Civets (Viverridae)',
        5: 'Colugos (Cynocephalidae)',
        6: 'Cuckoos (Cuculiformes)',
        7: 'Dogs (Canidae)',
        8: 'DragonLizards (Agamidae)',
        9: 'Frogs (Anura)',
        10: 'Geckos (Gekkonidae)',
        11: 'Kingfishers (Coraciiformes)',
        12: 'KitesHawksEagles (Accipitriformes)',
        13: 'Landfowls (Galliformes)',
        14: 'Monitors (Varanidae)',
        15: 'Monkeys (Cercopithecidae)',
        16: 'Mousedeers (Tragulidae)',
        17: 'Nightjars (Caprimulgiformes)',
        18: 'Otters (Mustelidae)',
        19: 'Owls (Strigiformes)',
        20: 'Pangolins (Manidae)',
        21: 'Passerines (Passeriformes)',
        22: 'PigeonsDoves (Columbiformes)',
        23: 'Rails (Gruiformes)',
        24: 'Rats (Muridae)',
        25: 'Shorebirds (Charadriiformes)',
        26: 'Shrews (Soricidae)',
        27: 'Skinks (Scincidae)',
        28: 'Snakes (Serpentes)',
        29: 'Squirrels (Sciuridae)',
        30: 'Storks (Ciconiiformes)',
        31: 'Treeshrews (Tupaiidae)',
        32: 'Turtles (Testudines)',
        33: 'Waterfowls (Anseriformes)',
        34: 'Woodpeckers (Piciformes)'
    }

    def __init__(self, weights=None, device="cpu"):
        """
        Initialize the Amazon animal Classifier.

        Args:
            weights (str, optional): Path to the model weights. Defaults to None.
            device (str, optional): Device for model inference. Defaults to "cpu".
        """

        super(SINClassifier, self).__init__(weights=weights, device=device,
                                            num_cls=35, num_layers=50)

    def results_generation(self, logits, img_ids, id_strip=None):
        """
        Generate results for classification.

        Args:
            logits (torch.Tensor): Output tensor from the model.
            img_id (str): Image identifier.
            id_strip (str): stiping string for better image id saving.       

        Returns:
            dict: Dictionary containing image ID, prediction, and confidence score.
        """
        
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1)[0]
        confidences = probs[0].tolist()
        result = [[i, confidence] for i, confidence in enumerate(confidences)]

        results = []
        for pred, img_id, conf in zip(preds, img_ids, confs):
            r = {"img_id": str(img_id).strip(id_strip)}
            r["prediction"] = self.CLASS_NAMES[pred.item()]
            r["class_id"] = pred.item()
            r["confidence"] = conf.item()
            r["all_confidences"] = result
            results.append(r)
        
        return results
