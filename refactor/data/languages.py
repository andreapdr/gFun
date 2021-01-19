"""
bg = Bulgarian
cs = Czech
da = Danish
de = German
el = Greek
en = English
es = Spanish
et = Estonian
fi = Finnish
fr = French
hu = Hungarian
it = Italian
lt = Lithuanian
lv = Latvian
nl = Dutch
mt = Maltese
pl = Polish
pt = Portuguese
ro = Romanian
sk = Slovak
sl = Slovene
sv = Swedish
"""

NLTK_LANGMAP = {'da': 'danish', 'nl': 'dutch', 'en': 'english', 'fi': 'finnish', 'fr': 'french', 'de': 'german',
                'hu': 'hungarian', 'it': 'italian', 'pt': 'portuguese', 'ro': 'romanian', 'es': 'spanish', 'sv': 'swedish'}


#top 10 languages in wikipedia order by the number of articles
#LANGS_10_MOST_WIKI = ['en','fr','sv','de','es','it','pt','nl','pl','ro']

#all languages in JRC-acquis v3
JRC_LANGS = ['bg','cs','da','de','el','en','es','et','fi','fr','hu','it','lt','lv','mt','nl','pl','pt','ro','sk','sl','sv']
JRC_LANGS_WITH_NLTK_STEMMING = ['da', 'nl', 'en', 'fi', 'fr', 'de', 'hu', 'it', 'pt', 'es', 'sv'] # 'romanian deleted for incompatibility issues'

RCV2_LANGS = ['ru', 'de', 'fr', 'sv', 'no', 'da', 'pt', 'it', 'es', 'jp', 'htw', 'nl']
RCV2_LANGS_WITH_NLTK_STEMMING = ['de', 'fr', 'sv', 'da', 'pt', 'it', 'es', 'nl']

lang_set = {'JRC_NLTK':JRC_LANGS_WITH_NLTK_STEMMING,   'JRC':JRC_LANGS,
            'RCV2_NLTK':RCV2_LANGS_WITH_NLTK_STEMMING, 'RCV2':RCV2_LANGS}

