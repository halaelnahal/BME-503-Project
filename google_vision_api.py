def requestLabel(fileName):
    import io
    import os

    # Imports the Google Cloud client library
    from google.cloud import vision
    from google.cloud.vision import types

    # Provide Google credential in script instead of one-time setup in environment
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "BME503-cd8d324b4f4b.json"

    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    file_name = os.path.join(os.path.dirname(__file__),
    fileName)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    for label in labels:
        primaryLabel = (label.description)
        primaryScore = (label.score)
        break
    return [primaryLabel, primaryScore]
