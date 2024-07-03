import csv
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns


stop_words = {
    'x', 'y', 'your', 'yours', 'yourself', 'yourselves', 'you', 'yond', 'yonder', 'yon', 'ye', 'yet', 'z',
    'zillion', 'j', 'u', 'umpteen', 'usually', 'us', 'username', 'uponed', 'upons', 'uponing', 'upon', 'ups',
    'upping', 'upped', 'up', 'unto', 'until', 'unless', 'unlike', 'unliker', 'unlikest', 'under',
    'underneath', 'use', 'used', 'usedest', 'r', 'rath', 'rather', 'rathest', 'rathe', 're', 'relate',
    'related', 'relatively', 'regarding', 'really', 'res', 'respecting', 'respectively', 'q', 'quite', 'que',
    'qua', 'n', 'neither', 'neaths', 'neath', 'nethe', 'nethermost', 'necessary', 'necessariest',
    'necessarier', 'never', 'nevertheless', 'nigh', 'nighest', 'nigher', 'nine', 'noone', 'nobody',
    'nobodies', 'nowhere', 'nowheres', 'no', 'noes', 'nor', 'nos', 'no-one', 'none', 'not', 'notwithstanding',
    'nothings', 'nothing', 'nathless', 'natheless', 't', 'ten', 'tills', 'till', 'tilled', 'tilling', 'to',
    'towards', 'toward', 'towardest', 'towarder', 'together', 'too', 'thy', 'thyself', 'thus', 'than', 'that',
    'those', 'thou', 'though', 'thous', 'thouses', 'thoroughest', 'thorougher', 'thorough', 'thoroughly',
    'thru', 'thruer', 'thruest', 'thro', 'through', 'throughout', 'throughest', 'througher', 'thine', 'this',
    'thises', 'they', 'thee', 'the', 'then', 'thence', 'thenest', 'thener', 'them', 'themselves', 'these',
    'therer', 'there', 'thereby', 'therest', 'thereafter', 'therein', 'thereupon', 'therefore', 'their',
    'theirs', 'thing', 'things', 'three', 'two', 'o', 'oh', 'owt', 'owning', 'owned', 'own', 'owns', 'others',
    'other', 'otherwise', 'otherwisest', 'otherwiser', 'of', 'often', 'oftener', 'oftenest', 'off', 'offs',
    'offest', 'one', 'ought', 'oughts', 'our', 'ours', 'ourselves', 'ourself', 'out', 'outest', 'outed',
    'outwith', 'outs', 'outside', 'over', 'overallest', 'overaller', 'overalls', 'overall', 'overs', 'or',
    'orer', 'orest', 'on', 'oneself', 'onest', 'ons', 'onto', 'a', 'atween', 'at', 'athwart', 'atop', 'afore',
    'afterward', 'afterwards', 'after', 'afterest', 'afterer', 'ain', 'an', 'any', 'anything', 'anybody',
    'anyone', 'anyhow', 'anywhere', 'anent', 'anear', 'and', 'andor', 'another', 'around', 'ares', 'are',
    'aest', 'aer', 'against', 'again', 'accordingly', 'abaft', 'abafter', 'abaftest', 'abovest', 'above',
    'abover', 'abouter', 'aboutest', 'about', 'aid', 'amidst', 'amid', 'among', 'amongst', 'apartest',
    'aparter', 'apart', 'appeared', 'appears', 'appear', 'appearing', 'appropriating', 'appropriate',
    'appropriatest', 'appropriates', 'appropriater', 'appropriated', 'already', 'always', 'also', 'along',
    'alongside', 'although', 'almost', 'all', 'allest', 'aller', 'allyou', 'alls', 'albeit', 'awfully', 'as',
    'aside', 'asides', 'aslant', 'ases', 'astrider', 'astride', 'astridest', 'astraddlest', 'astraddler',
    'astraddle', 'availablest', 'availabler', 'available', 'aughts', 'aught', 'vs', 'v', 'variousest',
    'variouser', 'various', 'via', 'vis-a-vis', 'vis-a-viser', 'vis-a-visest', 'viz', 'very', 'veriest',
    'verier', 'versus', 'k', 'g', 'go', 'gone', 'good', 'got', 'gotta', 'gotten', 'get', 'gets', 'getting',
    'b', 'by', 'byandby', 'by-and-by', 'bist', 'both', 'but', 'buts', 'be', 'beyond', 'because', 'became',
    'becomes', 'become', 'becoming', 'becomings', 'becominger', 'becomingest', 'behind', 'behinds', 'before',
    'beforehand', 'beforehandest', 'beforehander', 'bettered', 'betters', 'better', 'bettering', 'betwixt',
    'between', 'beneath', 'been', 'below', 'besides', 'beside', 'm', 'my', 'myself', 'mucher', 'muchest',
    'much', 'must', 'musts', 'musths', 'musth', 'main', 'make', 'mayest', 'many', 'mauger', 'maugre', 'me',
    'meanwhiles', 'meanwhile', 'mostly', 'most', 'moreover', 'more', 'might', 'mights', 'midst', 'midsts',
    'h', 'huh', 'humph', 'he', 'hers', 'herself', 'her', 'hereby', 'herein', 'hereafters', 'hereafter',
    'hereupon', 'hence', 'hadst', 'had', 'having', 'haves', 'have', 'has', 'hast', 'hardly', 'hae', 'hath',
    'him', 'himself', 'hither', 'hitherest', 'hitherer', 'his', 'how-do-you-do', 'however', 'how', 'howbeit',
    'howdoyoudo', 'hoos', 'hoo', 'w', 'woulded', 'woulding', 'would', 'woulds', 'was', 'wast', 'we', 'wert',
    'were', 'with', 'withal', 'without', 'within', 'why', 'what', 'whatever', 'whateverer', 'whateverest',
    'whatsoeverer', 'whatsoeverest', 'whatsoever', 'whence', 'whencesoever', 'whenever', 'whensoever', 'when',
    'whenas', 'whether', 'wheen', 'whereto', 'whereupon', 'wherever', 'whereon', 'whereof', 'where',
    'whereby', 'wherewithal', 'wherewith', 'whereinto', 'wherein', 'whereafter', 'whereas', 'wheresoever',
    'wherefrom', 'which', 'whichever', 'whichsoever', 'whilst', 'while', 'whiles', 'whithersoever', 'whither',
    'whoever', 'whosoever', 'whoso', 'whose', 'whomever', 's', 'syne', 'syn', 'shalling', 'shall', 'shalled',
    'shalls', 'shoulding', 'should', 'shoulded', 'shoulds', 'she', 'sayyid', 'sayid', 'said', 'saider',
    'saidest', 'same', 'samest', 'sames', 'samer', 'saved', 'sans', 'sanses', 'sanserifs', 'sanserif', 'so',
    'soer', 'soest', 'sobeit', 'someone', 'somebody', 'somehow', 'some', 'somewhere', 'somewhat', 'something',
    'sometimest', 'sometimes', 'sometimer', 'sometime', 'several', 'severaler', 'severalest', 'seriously',
    'serious', 'senza', 'sent', 'seem', 'seemed', 'seeminger', 'seemingly', 'seemingest', 'seemings', 'seems',
    'seven', 'save'
}

file = open('train.csv')
csvreader = csv.reader(file)
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)
file.close()


def preprocess(text):
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return words


all_preprocess = [preprocess(i) for i in [row[1] for row in rows]]


tokens = [i for j in all_preprocess for i in j]

frequency = {}
for i in tokens:
    if i in frequency:
        frequency[i] += 1
    else:
        frequency[i] = 1

final_features = sorted(frequency, key=frequency.get, reverse=True)[:1000]
print("Task 1")
print("30 most common words:")
for i in final_features[:30]:
    print(i)
print()

positive_reviews = []
negative_reviews = []

for row in rows:
    if row[0] == '5':
        positive_reviews.append(row[1])
    else:
        negative_reviews.append(row[1])

positive_reviews = [preprocess(review) for review in positive_reviews]
negative_reviews = [preprocess(review) for review in negative_reviews]


freq_d = {word: [0, 0] for word in final_features}
for review in positive_reviews:
    for word in review:
        if word in freq_d:
            freq_d[word][0] += 1

for review in negative_reviews:
    for word in review:
        if word in freq_d:
            freq_d[word][1] += 1


def train_nb(frequencies, positive_reviews, negative_reviews, final_features):
    loglikelihood = {}
    logprior = 0

    unique_words = set(final_features)
    k = len(unique_words)

    N_pos = N_neg = 0

    for word in final_features:
        N_pos += (frequencies[word][0])
        N_neg += (frequencies[word][1])

    logprior = math.log(len(positive_reviews)) - math.log(len(negative_reviews))

    for word in final_features:
        freq_pos = frequencies[word][0]
        freq_neg = frequencies[word][1]
        p_w_pos = (freq_pos + 1) / (N_pos + k)
        p_w_neg = (freq_neg + 1) / (N_neg + k)
        loglikelihood[word] = math.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_nb(freq_d, positive_reviews, negative_reviews, final_features)


def nb_predict(review, logprior, loglikelihood):
    words = preprocess(review)
    score = logprior
    for word in words:
        if word in loglikelihood:
            score += loglikelihood[word]
    return score


def evaluate_model(rows, logprior, loglikelihood):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in rows:
        rating, review = row[0], row[1]
        prediction = nb_predict(review, logprior, loglikelihood)
        if prediction > 0:
            if rating == '5':
                true_positive += 1
            else:
                false_positive += 1
        else:
            if rating == '5':
                false_negative += 1
            else:
                true_negative += 1

    accuracy = ((true_positive + true_negative) /
                (true_positive + true_negative + false_positive + false_negative))

    conf_matrix = [[true_negative,false_positive],[false_negative,true_positive]]

    return accuracy, conf_matrix


file = open('test.csv')
csvreader = csv.reader(file)
header = next(csvreader)

test = []
for row in csvreader:
    test.append(row)
file.close()


accuracy, conf_matrix = evaluate_model(test, logprior, loglikelihood)
print("Task 2")
print(f"Model Accuracy: {accuracy:.3f}")
print()
plt.figure(figsize=(8, 6), dpi=100)

ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', )
ax.set_xlabel("Predicted label", fontsize=14, labelpad=10)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.set_ylabel("True label", fontsize=14, labelpad=10)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
ax.set_title("Confusion Matrix", fontsize=14, pad=10)
plt.show()


def percentage_shuffle(rows, percentage):
    random.shuffle(rows)
    point = int(len(rows) * percentage/100)
    return rows[:point]


def learning_curve_analysis(train_data, test_data):
    percentages = [10, 30, 50, 70, 100]
    accuracies = []

    for percentage in percentages:
        subset = percentage_shuffle(train_data.copy(), percentage)
        positive_reviews = []
        negative_reviews = []

        for row in subset:
            if row[0] == '5':
                positive_reviews.append(row[1])
            else:
                negative_reviews.append(row[1])

        positive_reviews = [preprocess(review) for review in positive_reviews]
        negative_reviews = [preprocess(review) for review in negative_reviews]

        freq_d = {word: [0, 0] for word in final_features}
        for review in positive_reviews:
            for word in review:
                if word in freq_d:
                    freq_d[word][0] += 1

        for review in negative_reviews:
            for word in review:
                if word in freq_d:
                    freq_d[word][1] += 1

        logprior, loglikelihood = train_nb(freq_d, positive_reviews, negative_reviews, final_features)
        accuracy,conf_matrix = evaluate_model(test_data, logprior, loglikelihood)
        accuracies.append(accuracy)
        print(f"Training data used: {percentage}%, Accuracy: {accuracy:.3f}")

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot([i for i in percentages], accuracies, marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Training data used (%)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

print("Task 3")
learning_curve_analysis(rows, test)
print()
