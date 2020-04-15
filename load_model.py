from zoo.pipeline.inference import InferenceModel


def load_model(model_path, weight_path, batch_size):
    model = InferenceModel(supported_concurrent_num=1)
    model.load_openvino(model_path=model_path,
                        weight_path=weight_path,
                        batch_size=batch_size)
    return model
