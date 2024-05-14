import numpy as np


def unconditional():
    return "A PAS stained slide of a piece of kidney tissue"

def generate_normals(means, stddev):
    samples = []
    for m in means:
        samples.append(max(0, np.random.normal(m, stddev)))

    return samples


class CaptionGenerator():
    def generate(self):
        return ""
class RatKidneyConditional(CaptionGenerator):
    # Generate probabilities randomly if not given
    # The average probabilities in the rat-tissue dataset:
    # Readable:  [0.0037, 0.0001, 0.0271, 0.4418, 0.0121, 0.0219, 0.376, 0.1172]

    def generate(self, probabilities = None):
        if probabilities is None:
            averages = [3.72632114e-03, 1.39337593e-04, 2.70518503e-02, 4.41828884e-01, 1.21383491e-02, 2.18519807e-02,
                       3.76029002e-01, 1.17234275e-01]
            stddev = 0.1
            probabilities = generate_normals(means=averages, stddev=stddev)

        classes = ['White background, should be ignored', 'Arteries', 'Atrophic Tubuli', 'Tubuli', 'Glomeruli',
                   'Sclerotic Glomeruli', 'other kidney tissue',
                   'Dilated Tubuli']
        thresholds = {'low': 0.2, 'medium': 0.4}  # Define thresholds for low, medium, and high prevalence

        caption = "This image showcases various types of tissue structures found in renal tissue. \n"
        for i, prob in enumerate(probabilities):
            if i == 0:
                continue  # First value should be ignored
            if prob == 0:
                continue  # Skip classes with zero probability
            elif prob < thresholds['low']:
                caption += f"The image shows a low amount of {classes[i]}.\n"
            elif prob < thresholds['medium']:
                caption += f"The prevalence of {classes[i]} is medium.\n"
            else:
                caption += f"There is a lot of {classes[i]} visible in the image.\n"

        return caption


class Glomeruli(RatKidneyConditional):
    def generate(self):
        # Always returns the same caption, where tubuli are medium, glomeruli are high and the rest is 0
        return super().generate([0,0,0,0.3,1,0,0])