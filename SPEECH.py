import speech

def response(phrase, listener):
    speech.say("You said %s" % phrase)
    if phrase == "turn off":
        listener.stoplistening()
