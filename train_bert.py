from transformers import pipeline

# Load pretrained fake news classifier
classifier = pipeline("text-classification", 
                      model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Test
text = "Scientists say drinking petrol cures all diseases instantly"

result = classifier(text)

print(result)