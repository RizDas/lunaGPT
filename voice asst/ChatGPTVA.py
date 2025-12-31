def SpeakText(a):
    return "hello world"

messages = []
while(1):
    text = "lorem"
    messages.append({"role": "user", "content": text})
    response = "ipsum"
    SpeakText(response)

    print(response)