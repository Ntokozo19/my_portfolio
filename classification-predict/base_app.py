"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


# Data dependencies
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt







# Vectorizer
news_vectorizer = open("resources/tfidf_vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer)# loading your vectorizer from the pkl file

# Load your raw data
#raw = pd.read_csv("resources/train.csv")

DATA = ("resources/train.csv")
@st.cache(persist=True, allow_output_mutation=True)
def load_data():
    data = pd.read_csv(DATA)

    return data

data = load_data()

def main():
    st.title("Sentiments")
    st.markdown("by Team TS4")
    st.markdown("### Sentiment Analysis of Tweets about climate change")
    st.image('resources/twitter-logo.jpg')
    st.markdown("This App is a dashboard used "
                "to analyze sentiments of tweets about climate change and to make predictions of the sentiment "
                "based on a tweet.")
    st.markdown("### Use the side bar to explore the data and make predictions")
    st.sidebar.title("Analysis of Tweet Sentiments")

    #Display # of tweets by sentiment
    st.sidebar.markdown("### View the number of tweets by sentiment")
    select = st.sidebar.selectbox('Choose the Visualization type', ['Bar plot', 'Pie chart'], key='1')
    sentiment_count = data['sentiment'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
    if not st.sidebar.checkbox("Hide", True):
        st.markdown("### Number of tweets by sentiment")
        if select == 'Bar plot':
            fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
            st.plotly_chart(fig)
            st.markdown("1: The tweet supports the belief of man-made climate change")
            st.markdown("-1: The tweet does not believe in man-made climate change")
            st.markdown("0: The tweet neither supports nor refutes the belief of man-made climate change")
            st.markdown("2: The tweet links to factual news about climate change")
            st.markdown("Taking a closer look at the distribution of the tweets we notice that the data is severely "
                        "imbalanced with the majority of tweets falling in the 'pro' category, supporting the belief "
                        "of 'man-made climate change.")
        else:
            fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
            st.plotly_chart(fig)
            st.markdown("1: The tweet supports the belief of man-made climate change")
            st.markdown("-1: The tweet does not believe in man-made climate change")
            st.markdown("0: The tweet neither supports nor refutes the belief of man-made climate change")
            st.markdown("2: The tweet links to factual news about climate change")
            st.markdown("Taking a closer look at the distribution of the tweets we notice that the data is severely "
                        "imbalanced with the majority of tweets falling in the 'pro' category, supporting the belief "
                        "of 'man-made climate change.")

    #Frquent words word cloud
    st.sidebar.markdown(" ### Frequent Words used for each sentiment")
    word_sentiment = st.sidebar.radio('Which sentiment would you like to view?', ('Pro', 'Neutral', 'Anti', 'News'))
    news = data[data['sentiment'] == 2]['message']
    pro = data[data['sentiment'] == 1]['message']
    neutral =data[data['sentiment'] ==0]['message']
    anti = data[data['sentiment'] ==-1]['message']
    if not st.sidebar.checkbox("Hide", True, key=1):
        if word_sentiment == 'Pro':
            st.markdown("### Frequent words used in Pro climate change tweets")
            pro = [word for line in pro for word in line.split()]
            pro = WordCloud(
                background_color='white',
                max_words=50,
                max_font_size=100,
                scale=5,
                random_state=1,
                collocations=False,
                normalize_plurals=False
            ).generate(' '.join(pro))
            fig, ax = plt.subplots()
            ax.imshow(pro)
            plt.xticks([])
            plt.yticks([])
            st.pyplot(fig)
            st.markdown("We see words like believe, combat, fight, real and action which represent the pro "
                        "climate change supporters who believe that climate change is real and that action "
                        "needs to be taken stop it.")
            st.markdown("'https' occurs frequently in pro climate change tweets, implying that many links are being "
                        "shared around the topic of climate change. These could be links to petitions, websites "
                        "and/or articles related to climate change.")

        if word_sentiment == 'Anti':
            st.markdown("### Frequent words used in Anti climate change tweets")
            anti = [word for line in anti for word in line.split()]
            anti = WordCloud(
                background_color='white',
                max_words=50,
                max_font_size=100,
                scale=5,
                random_state=1,
                collocations=False,
                normalize_plurals=False
            ).generate(' '.join(anti))
            fig, ax = plt.subplots()
            ax.imshow(anti)
            plt.xticks([])
            plt.yticks([])
            st.pyplot(fig)
            st.markdown("Anti climate change tweets contain words such as 'hoax', 'scam', 'tax', 'liberal' and 'fake'. "
                        "Which drives home the fact the anti climate change tweeps believe that climate change is not "
                        "real.")
            st.markdown("'Trump' is a frequently occuring word in all 4 classes. During his presidency Donald trump "
                        "did not shy away from expressing his anti climate change views and his supporters didn't either."
                        " That is why Trump is also one of the most frequently used words in the anti climate change "
                        "tweets.")

        if word_sentiment == 'Neutral':
            st.markdown("### Frequent words used in Neutral climate change tweets")
            neutral = [word for line in neutral for word in line.split()]
            neutral = WordCloud(
                background_color='white',
                max_words=50,
                max_font_size=100,
                scale=5,
                random_state=1,
                collocations=False,
                normalize_plurals=False
            ).generate(' '.join(neutral))
            fig, ax = plt.subplots()
            ax.imshow(neutral)
            plt.xticks([])
            plt.yticks([])
            st.pyplot(fig)
            st.markdown("A vast majority of the neutrals are discussing, engaging and asking about the effects on "
                        "climate change as seen with words like 'talk','debate','report'. They are speaking about the "
                        "penguins which are endangered due to the effects of climate change. They speak about the "
                        "climate, weather, warm, polar bear. There also appears to be http, which suggests a lot of "
                        "neutral tweets may have a link to an article.")

        if word_sentiment == 'News':
            st.markdown("### Frequent words used in News climate change tweets")
            news = [word for line in news for word in line.split()]
            news = WordCloud(
                background_color='white',
                max_words=50,
                max_font_size=100,
                scale=5,
                random_state=1,
                collocations=False,
                normalize_plurals=False
            ).generate(' '.join(news))
            fig, ax = plt.subplots()
            ax.imshow(news)
            plt.xticks([])
            plt.yticks([])
            st.pyplot(fig)
            st.markdown("As we can see, the vast majority of the news is reporting about Donald Trump and his "
                        "views on climate change. They also report a vast majority of issues all included in the "
                        "analysis of the other words included. US Administrator of Environmental Affairs is appearing "
                        "regularly as well. There also is an 'http', which shows that news tweets may have a link to "
                        "a news report. The word cloud is also indicating that the words are well distributed and "
                        "spoken of almost similarly. Most of the words are not frequently requiring except "
                        "'Climate Change', 'Change HTTP', 'Warm HTTP', which are linked to tweets having links, and "
                        "the main topic being that of global warming and climate change.")

    #hashtag analysis
    st.sidebar.markdown(" ### Hashtags used for each sentiment")
    hashtag_sentiment = st.sidebar.radio('Which sentiment would you like to view?', ('Pro', 'Neutral', 'Anti', 'News'),key=6)
    if not st.sidebar.checkbox("Hide", True, key=5):
        if hashtag_sentiment == 'Pro':
            st.markdown("### Frequent hashtags used in Pro climate change tweets")
            st.image('resources/pro_hashtags.png')
            st.markdown("1. The most used pro climate change hastag outside of the #climate is #beforetheflood. "
                        "This hashtag comes from the documentary named Before the flood where the famous actor "
                        "Leonardo DiCaprio meets with scientists, activists and world leaders to discuss the dangers "
                        "of climate change and possible solutions. This documentary created a lot of awareness and "
                        "education around climate change and the causes of it. It brought, to everyday people's TV "
                        "screens, the truth about climate change and the true impact it has and will continue to have "
                        "on the planet if we dont change our way of living. Since this show, many people have jumped "
                        "on the pro climate change band wagon and have started speaking out, which is what we are "
                        "seeing with this hashtag.")
            st.markdown("2. Another famous hashtag in the pro climate change hashtags is the #ImVotingBecause. "
                        "As Americans select their next president, voters share their thoughts as to why they have "
                        "chosen their candidate with the hashtag #ImVotingBecause. The social media landscape shows "
                        "that supporters are firm in their convictions and consider this election a historic one. "
                        "This hashtag also falls under climate change issues because in America climate change has "
                        "become one of the most important issues in politics and climate change supporters want to "
                        "vote for a president who views climate change as a priority. Donald Trump was also a "
                        "president who reversed a lot of work done by climate change advocates around the world.")
            st.markdown("3. Another hashtag that was in the top 5 was the #COP22 which is The 2016 United Nations "
                        "Climate Change Conference that was an international meeting of political leaders and "
                        "activists to discuss environmental issues. It was held in Marrakech, Morocco, on 7–18 "
                        "November 2016. This is where the Trump administration formerly announced their plans to "
                        "exit the climate change deal. Climate change supporters where not happy with this decision. "
                        "This also exlains the #Trump which also appears at number 9 of the pro climate change "
                        "hashshtags.")

        if hashtag_sentiment == 'Anti':
            st.markdown("### Frequent hashtags used in Anti climate change tweets")
            st.image('resources/anti_hashtags.png')
            st.markdown("1. The number hastag one used by anti climate changers is #MAGA. 'Make America Great Again' or "
                        "MAGA is a campaign slogan used in American politics popularized by Donald Trump in his "
                        "successful 2016 presidential campaign. During his presidency Donald trump did not shy away "
                        "from expressing his anti climate change views and his supporters didnt either. That is why "
                        "#Trump is also the 3rd most frequently used hashtag in the anti climate change hashtags.")
            st.markdown("2. Hashtags such as #Fakenews, #climatescam, #DrainTheSwamp alsomade it to the top hastagsand "
                        "they are all related to anti climate chmage supporters who believe thet climate chnage is a "
                        "lie or a scam which is also perpurtuated by Donald Trump. So we can see from all these "
                        "hashtags that Donald Trump is the biggest leader who is anti climate change and who's "
                        "supporters are also anti climate change.")

    #length of tweet analysis
    st.sidebar.markdown(" ### Length of tweet per sentiment")
    if not st.sidebar.checkbox("Hide", True, key=7):
        st.markdown("### Length of tweet per sentiment")
        st.image('resources/length_of_tweet.png')
        st.markdown("From the above we can see that the tweets that represent the pro climate change group are "
                    "generally longer than the other sentiments, meaning people that are pro climate change write "
                    "longer tweets as compared to the other groups. We can also see that people who are against "
                    "climate change generally write shorter tweets as compared to the pro and neutral groups.")

#Prediction

    st.sidebar.markdown(" ### Make Predictions")
    model_choice = st.sidebar.radio('Which model would you like to use?', ('Logistic Regression', 'Linear SVC', 'KNearest Neighbors'),key=9)
    if not st.sidebar.checkbox("Hide", True, key=8):
        if model_choice == 'Logistic Regression':
            st.markdown("### Make a prediction of the sentiment of a tweet")
            tweet_text = st.text_area("Enter tweet to get a prediction", "Type Here")
            if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
                        prediction = predictor.predict(vect_text)


                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success("Tweet Classified as: {} where 0 = Neutral, 1 = Pro Climate Change, "
                                   "-1 = Anti Climate change, 2 = News".format(prediction))

        if model_choice == 'Linear SVC':
            st.markdown("### Make a prediction of the sentiment of a tweet")
            tweet_text = st.text_area("Enter tweet to get a prediction", "Type Here")
            if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(open(os.path.join("resources/lsvc.pkl"),"rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success("Tweet Classified as: {} where 0 = Neutral, 1 = Pro Climate Change, "
                                   "-1 = Anti Climate change, 2 = News".format(prediction))


        if model_choice == 'KNearest Neighbors':
            st.markdown("### Make a prediction of the sentiment of a tweet")
            tweet_text = st.text_area("Enter tweet to get a prediction", "Type Here")
            if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(open(os.path.join("resources/knn.pkl"),"rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success("Tweet Classified as: {} where 0 = Neutral, 1 = Pro Climate Change, "
                                   "-1 = Anti Climate Change, 2 = News".format(prediction))

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
