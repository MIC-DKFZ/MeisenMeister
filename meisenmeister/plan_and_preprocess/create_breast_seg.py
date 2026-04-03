from breastdivider import BreastDividerPredictor


def predict_breast_segmentation(input_path, output_path):
    predictor = BreastDividerPredictor(device="cuda")
    predictor.predict(
        input_path=input_path,
        output_path=output_path,
    )
