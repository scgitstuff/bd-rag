# bd-rag

boot.dev "Retrieval Augmented Generation" project

# dependency

```shell
uv add nltk==3.9.1
# I have an old video card, need older torch
uv add sentence-transformers torch==2.6
uv add numpy
```

# data files

the lesson had links, the files are not checked in  
https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json  
https://en.wikipedia.org/wiki/Stop_word

```
data/
    movies.json
    stopwords.txt
```
