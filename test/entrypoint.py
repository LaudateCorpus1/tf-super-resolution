from ai_integration.public_interface import start_loop

from inference import initialize_model, infer

initialize_model()

start_loop(inference_function=infer, inputs_schema={
    "image": {
        "type": "image"
    }
} )