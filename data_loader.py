import csv
from preprocessing import preprocess


EMOTION_CLASSES = ["happiness", "anger", "sadness", "neutral"]


class DataLoader:
    @staticmethod
    def load_crowdflower_db(path):
        """
        Loads CrowdFlower database. Database contains of 14 classes:
        worry, enthusiasm, sadness, love, anger, surprise, relief, sentiment, happiness, fun, boredom, hate, neutral

        HAPPINESS:  enthusiasm, happiness, fun
        ANGER:      anger, hate
        SADNESS:    sadness
        NEUTRAL:    neutral

        :param path: path to database
        :return: Dataset object"""

        emotion_map = {"enthusiasm": "happiness", "happiness": "happiness", "fun": "happiness",
                       "sadness": "sadness", "anger": "anger","hate": "anger", "neutral": "neutral"}

        dataset = {class_name: [] for class_name in EMOTION_CLASSES}
        with open(path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(spamreader):
                # row is a list of 4 elements: tweet_id, emotion, user, content(may be seperated if contains ",")
                emotion = row[1]
                emotion = emotion.strip("\"")
                content = ",".join(row[3:])
                content = content.strip("\"")
                content = preprocess(content)
                # import pdb;pdb.set_trace()
                if emotion in emotion_map.keys():
                    dataset[emotion_map[emotion]].append(content)
        return dataset




