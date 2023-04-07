from func import *

print('this is da MAIN')

for topic in list_of_topics:
    retrieve_documents(topic)
  
print(set_of_all_documents)

## The following line took 36 minutes to run:
rv = get_all_relevant_documents("Who is the President of the US?")

for doc in rv["Who is the President of the US?"]:
    print(doc.link, doc.text)

print(rv["question"][0].link, rv["question"][0].text)