import re

numbers = "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand)"
day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
week_day = "(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday| \
            thursday|friday|saturday|sunday)"
month = "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february| \
         march|april|may|june|july|august|september|october|november|december)"
dmy = "(year|day|week|month)"
rel_day = "(today|yesterday|tomorrow|tonight|tonite)"
exp1 = "(before|after|earlier|later|ago)"
exp2 = "(until|in|on|this|next|last)"
iso = "\d+[/-]\d+[/-]\d+ \d+:\d+:\d+\.\d+"
year = "((?<=\s)\d{4}|^\d{4})"
regxp1 = "((\d+|(" + numbers + "[-\s]?)+) " + dmy + "s? " + exp1 + ")"
regxp2 = "(" + exp2 + " (" + dmy + "|" + week_day + "|" + month + "))"
regxp3 = ("(" + exp2 + "?\W+" + month + " " + numbers + "(\.|st|nd|rd|th)\W+" +
          year + "?)")
regxp4 = ("(" + exp2 + "\W+((" + month + "\W+" + year + ")|" + month + "|" +
          year + "))")
regxp5 = "(" + day + ")"

reg = [
    re.compile(regxp1, re.IGNORECASE),
    re.compile(regxp2, re.IGNORECASE),
    re.compile(regxp3, re.IGNORECASE),
    re.compile(regxp4, re.IGNORECASE),
    re.compile(regxp5, re.IGNORECASE),
    re.compile(rel_day, re.IGNORECASE),
    re.compile(iso),
    re.compile(year)
]


def tag(word_list):

    word_beginning = [0]
    accumulator = 0
    for word in word_list:
        accumulator += len(word) + 1
        word_beginning.append(accumulator)

    sentence = " ".join(word_list)
    tagged_list = [False] * len(word_list)

    for r in reg:
        for m in r.finditer(sentence):
            for i, beg in enumerate(word_beginning):
                if beg >= m.start() and beg <= m.end():
                    tagged_list[i] = True

    return tagged_list
