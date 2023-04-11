# LangChain-Gradio Template


## To run: 
```
gradio appp.py 
```

This repo serves as a template for how to deploy a LangChain on Gradio.
This is particularly useful because you can easily deploy Gradio apps on Hugging Face spaces, 
making it very easy to share you LangChain applications on there.

This repo contains an `app.py` file which has a template for a chatbot implementation.
This was heavily inspired [James Weaver's examples](https://huggingface.co/JavaFXpert).

## Adding your chain
To add your chain, you need to change the `load_chain` function in `app.py`.
Depending on the type of your chain, you may also need to change the inputs/outputs that occur later on (in the `chat` function).

## Deploying to Hugging Face spaces
To deploy to Hugging Face spaces, you should first create a new Gradio space on Hugging Face.
From there, you can then copy over the `app.py` file here, and the `requirements.txt`.
Do not copy over this ReadMe - Hugging Face spaces requires a ReadMe in a particular format.
