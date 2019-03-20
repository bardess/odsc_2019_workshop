import re
import spacy
from bs4.element import Tag
from collections import namedtuple
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import stop_words


Sentence = namedtuple('Sentence', ['sid', 'section', 'title', 'text', 'subj', 'v', 'obj'])

SUBJECTS = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl']

OBJECTS = ['dobj', 'dative', 'attr', 'oprd']

BURDENS = ['shall', 'must', 'ought', 'oblige', 'require']

STOPWORDS = set(
    list(stop_words.ENGLISH_STOP_WORDS) +\
    stopwords.words('english') +\
    ['mm', 'section', 'subsection', 'schedule', '-PRON-']
)

# fidn references to laws e.g. O.Reg. 191/11, s.4

nlp = spacy.load('en')

ref_str = '(O. ?Reg. ?[0-9]+/[0-9]+)(,?\.? ?[sS](chedule)?\.? [0-9]+)+( \([0-9]+\))?\.|([0-9]{4},) (c. [0-9]+)(, Sched. [A-Z]+)?(, s. [0-9]+)?( \([0-9]+\))?\.'

LAW_REGEX = re.compile(ref_str)


class LemmaTokenizer(object):
    
    def __call__(self, text):
        
        return [token.lemma_ for token in nlp(text) 
                if token.is_alpha and token.lemma_ not in STOPWORDS]
    

def get_headnotes(html, prefix, tag='headnote-e'):

    headnotes = {}

    for h in html.find_all(class_=tag):

        title = h.text.replace('\xa0', '').strip()
        tags = []

        for sib in h.next_siblings:

            if isinstance(sib, Tag):

                if sib.has_attr('class') and\
                    any([cl.startswith(t) for cl in sib.attrs['class']
                         for t in ['headnote', 'schedule']]):
                
                    break

                else:

                    tags.append(sib)

        full_text = ' '.join([item.text.strip().replace('\xa0', ' ') for item in tags])

        for ref in re.finditer(LAW_REGEX, full_text):
            full_text = full_text.replace(ref.group(), '')

        full_text = re.sub(
            re.compile('(^)?([0-9]{1,2}\. )|(\([0-9]{1,2}(\.[0-9]{1,2})?\))|([ivx]+\.)|(\([a-z]\) )'),
            '', full_text).strip()

        headnotes['{0} {1}'.format(prefix, title)] = full_text

    return headnotes


def get_breadth_first_nodes(root):
    nodes = []
    stack = [root]
    while stack:
        cur_node = stack[0]
        stack = stack[1:]
        nodes.append(cur_node)
        for child in cur_node.children:
            stack.append(child)
    return nodes


def get_all(sent, tags):
    all_tokens = [token for token in sent if token.dep_ in tags]
                  #and token.head.dep_ == 'ROOT']
    bag_of_words = []
    for token in all_tokens:
        bag_of_words += get_breadth_first_nodes(token)
    return ' '.join([t.text for t in sorted(bag_of_words, key=lambda t: t.i)])


def make_sentence(sent_id, sent, section, title):
    verbs = [token.head if token.tag_ == 'MD' else token
             for token in sent if token.lemma_ in BURDENS]
    return Sentence(
        sid=sent_id,
        section=section,
        title=title,
        text=sent.text,
        subj=get_all(sent, SUBJECTS),
        v=verbs,
        obj=get_all(sent, OBJECTS)
    )


def vocab_barplot(df, label, max_rank=10):    
    
    subset = df[(df['rank'] <= max_rank) & (df.label == label)]
    
    plt.figure(figsize=(15,5))
    sns.barplot(data=subset, x='word', y='count', hue='label')
    plt.xticks(rotation=50)
    plt.xlabel('Word')
    plt.ylabel('Number of Occurrences')
    plt.show()
