"""
BT3102 Group Fifteen
Lee Shan Shan	A0241238L
Zhang Zishuo	A0239837M
Yang Haoying	A0238149X
Lim Wei Xuan	A0233827B

Naive prediction accuracy:     1071/1378 = 0.7772133526850508
Naive prediction2 accuracy:    1072/1378 = 0.7779390420899854

Viterbi prediction accuracy:   1056/1378 = 0.7663280116110305
Viterbi2 prediction accuracy:  1093/1378 = 0.793178519593614
"""

ddir = '/Users/yanghaoying/Desktop/Y2S2/BT3102/Project/project-files'

# Q2 and Q3 are left with a pass and commented out in run()
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    pass

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename,
):
    pass

# Q4 ------------------------------------------------------------------------------------------------------------------------------------

# Helper function to compute the output probabilities using train data
def compute_output(train_file, output_file):
    # Token = word, Tag = twitter_tags.txt

    # Create a dictionary of dictionaries to store the counts of a word being associated to a specific tag
    # key: token. value: {tag: no.of times token has tag}
    token_tag_counts = {}
    # Create dictionary to store the counts of each tag for denominator
    # key: tag. value: no.of times tag appears in training text
    tag_counts = {}

    # Open the training_file and split the tokens and tag
    with open(train_file, "r", encoding="utf8") as train:
        for line in train:
            # If not end of tweet
            if line.strip():
                token, tag = line.split("\t")
                tag = tag.strip()

                # If tokens / tags not inside their respective dictionaries
                if token not in token_tag_counts:
                    # if token not inside the token_tag_counts dict, add it in with count = 0
                    token_tag_counts[token] = {}

                if tag not in tag_counts:
                    # if tag is not already in the tag_counts dictionary, add it with count = 0
                    tag_counts[tag] = 0

                # Increment the count of the tag in the tag_counts dict
                tag_counts[tag] += 1

                # If the tag is not a key in the dict for token, add it with count = 0
                if tag not in token_tag_counts[token]:
                    token_tag_counts[token][tag] = 0

                # Increment the count of the tag for the token in the token_tag_counts dict
                token_tag_counts[token][tag] += 1

    # Dictionary of dictionary to store the output probabilties
    # key: token. value: {tag: P(token|tag)=count(token, tag)/count(tag)}
    output_probs = {}

    # For loop through each token in the token_tag_counts dictionary
    for token in token_tag_counts:
        # Creating a dict to store the output probabilties for this token.
        output_probs[token] = {}
        # Loop through each tag for this token in the token_tag_counts dict and use MLE
        for tag in token_tag_counts[token]:
            # numerator = number of times token w is associated with tag j
            # denominator = number of times tag j appears
            # Using smoothing constant = 1
            output_probs[token][tag] = (token_tag_counts[token][tag] + 1) / (
                tag_counts[tag] + 1 * (len(token_tag_counts) + 1)
            )

    # Writing to output file
    with open(output_file, "w", encoding="utf-8") as output:
        for token in output_probs:
            for tag in output_probs[token]:
                output.write(f"{token}\t{tag}\t{output_probs[token][tag]}\n")
                """Sample Output
                    RT	~	0.03445525602169405
                    RT	N	0.00025342118601115053
                """

# Helper function to compute the transition probabilities using train data
def compute_trans(train_file, tags_file, output_file):
    # initialize transition count dictionary to count the number of transitions between different pos tags.
    transition_counts = {}

    # initialize tag count dictionary to count number of times a tag appears.
    tag_counts = {}

    # Read in twitter_tags.txt
    with open(tags_file, "r", encoding="utf-8") as f:
        tags = [tag.strip() for tag in f.readlines()]
        # print(tags)

    # Open the training_file and split the tokens and tag and fill up both dictionaries.
    with open(train_file, "r", encoding="utf8") as train:
        # Set previous tag to = START for first.
        previous_tag = "START"
        for line in train:
            # If not end of tweet
            if line.strip():
                token, tag = line.split("\t")
                tag = tag.strip()

                if previous_tag == "START":
                    if "START" not in tag_counts:
                        tag_counts["START"] = 0
                    tag_counts["START"] += 1

                if previous_tag != None:
                    # Set transition to a tuple of previous_tag and current tag.
                    trans = (previous_tag, tag)
                    if trans not in transition_counts:
                        # If trans does not exist in transition_counts, add to transition_counts dictionary
                        transition_counts[trans] = 0
                    transition_counts[trans] += 1

                # If tag does not exist in tag counts, add to tagcounts dictionary
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1
                # Set current tag to previous_tag to prepare for next iteration.
                previous_tag = tag

            # if end of existing tweet, set previous tag back to START.
            else:
                trans = (previous_tag, "STOP")
                if trans not in transition_counts:
                    # If trans does not exist in transition_counts, add to transition_counts dictionary
                    transition_counts[trans] = 0
                transition_counts[trans] += 1
                # If tag does not exist in tag counts, add to tagcounts dictionary
                if "STOP" not in tag_counts:
                    tag_counts["STOP"] = 0
                tag_counts["STOP"] += 1
                # Set tag back to start for next iteration.
                previous_tag = "START"

    # Compute the transition probabilities with the 2 dictionaries.
    transition_probs = {}
    smoothing_const = 10
    num_tags = len(tags)  # should = 25
    # print("number of pos tags: " + str(num_tags))
    for transition, count in transition_counts.items():
        prev_tag = transition[0]  # first element of tuple
        # smoothing transitionprobability
        prob = (count + smoothing_const) / (tag_counts[prev_tag] + smoothing_const * (num_tags + 1))
        transition_probs[transition] = prob

    # Set any transition probabilities that do not occur in the training data to smoothed value
    tags.append("START")
    tags.append("STOP")
    # print(tags)
    for tag1 in tags:
        for tag2 in tags:
            if (tag1, tag2) not in transition_probs:
                transition_probs[(tag1, tag2)] = smoothing_const / (tag_counts[tag1] + smoothing_const * (num_tags + 1))

    # write the transition probabilities generated to trans_probs.txt
    with open(output_file, "w", encoding="utf-8") as f:
        for transition, prob in transition_probs.items():
            f.write(f"{transition[0]}\t{transition[1]}\t{prob:.6f}\n")
            """Sample Output
                ~	@	0.466038
                @	~	0.242804
                ~	O	0.054717
            """


# Generating output_probs.txt and trans_probs.txt for viterbi
compute_output(f"{ddir}/twitter_train.txt", "output_probs.txt")
compute_trans(f"{ddir}/twitter_train.txt", f"{ddir}/twitter_tags.txt", "trans_probs.txt")

# Helper function for using viterbi algorithm to predict most likely sequence of tags for a list of words (a tweet)
def viterbi_tweet(tweet, tags, output_probs, trans_probs):
    #Initialise n * (N-1) matrix for Pi as a dictionary of dictionary
    #key: k, value: {tag : prob(tag at time k)}
    Pi = [{}]

    #Initialise n * (N-1) matrix for BP as a dictionary of dictionary
    #key: k, value: {tag : list of tags from start till t}
    BP = [{}]

    n = len(tweet)

    #Base case: start -> each tag
    Pi.append({})
    BP.append({})
    x1 = tweet[0]
    for tag in tags:
        output_probs_tag_x1 = 0
        if (tag in output_probs):
            if (x1 in output_probs[tag]):
                output_probs_tag_x1 = output_probs[tag][x1]              
            else:
                output_probs_tag_x1 = 1e-10
        Pi[1][tag] = trans_probs["START"][tag] * output_probs_tag_x1
        BP[1][tag] = [tag]

    #Recursive
    for k in range(2, n+1):
        Pi.append({})
        BP.append({})
        xk = tweet[k-1]

        for v in tags:
            output_probs_v_xk = 0
            if (v in output_probs):
                if (xk in output_probs[v]):
                    output_probs_v_xk = output_probs[v][xk]
                else:
                    output_probs_v_xk = 1e-10
            (prob, prev_state) = max((Pi[k-1][u] * trans_probs[u][v] * output_probs_v_xk, u) for u in tags)
            Pi[k][v] = prob
            BP[k][v] = BP[k-1][prev_state] + [v]

    #Final
    (maxProb, finalBP) = max((Pi[n][v] * trans_probs[v]["STOP"], v) for v in tags)

    return BP[n][finalBP]
    

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):

    #Read all tags as a list
    with open(in_tags_filename, 'r', encoding="utf-8") as in_tags:
        #tags: a list containing 25 tags
        tags = [tag.strip() for tag in in_tags.readlines()]
    
    #Read the output probabilities from output_probs.txt
    with open(in_output_probs_filename, 'r', encoding="utf-8") as output:
        #output_probs: a dictionary of dictionary
        #key:token. value:{tag:P(token|tag)}.
        output_probs = {}
        for line in output: 
            token, tag, prob = line.strip().split('\t')

            if tag not in output_probs: 
                output_probs[tag] = {}
            
            output_probs[tag][token] = float(prob)

    #Read the transition probabilities from trans_probs.txt
    with open(in_trans_probs_filename, 'r', encoding="utf-8") as trans:
        #trans_probs: a dictionary of dictionary
        #key:tag i. value:{tag j: P(i -> j|tag i)}.
        trans_probs = {}
        for line in trans: 
            curr_tag, next_tag, prob = line.strip().split('\t')

            if curr_tag not in trans_probs: 
                trans_probs[curr_tag] = {}
            
            trans_probs[curr_tag][next_tag] = float(prob)



    with open(in_test_filename, 'r', encoding="utf-8") as inputs, open(out_predictions_filename, 'w', encoding="utf-8") as outputs:
        #For loop each line in input file

        curr_tweet = []
        output_predictions = []
        for line in inputs : 
            #If not end of tweet
            #This word is connected with the previous word and they belong to one tweet. Write into the current tweet as a list
            if line.strip() : 
                token = line.strip()
                curr_tweet.append(token)
            #If end of tweet
            #Current tweet ends here. This whole tweet should then be processed using the Viterbi Algorithm
            #Reset the curr_tweet list to empty for processing next tweet
            else:
                curr_tweet_predictions = viterbi_tweet(curr_tweet, tags, output_probs, trans_probs)
                output_predictions += curr_tweet_predictions
                output_predictions.append(" ")
                curr_tweet = []

        for predict_tag in output_predictions:
            #Write to output file
            outputs.write(f"{predict_tag}\n")


#Q5-----------------------------------------------------------------------------------------------------------------------------------------------
'''
Improvements : 
1. Preprocessing of training data before generating output_probs, and preprocessing of test data before prediction. 
- Processing Steps:
    a. Grouping of all @USER____  to @USER. 
    b. Grouping of all http://links to http://
- Benefits:
    a. Reduce the size of vocabulary for output_probs.
    b. Generalise better to new users and links for more accurate prediciton on unseen test data.
- Helper function for this improvement:
    preprocessing()
'''
# Import packages for replacing input text
import re

# Helper function for preprocessing according to the listed improvements:
def preprocessing(train_file, processed_train_file): 
    ''' Takes in twitter_train.txt and returns twitter_train_proc.txt '''
    # Open the training_file and split the tokens and tag
    with open(train_file, "r", encoding="utf8") as train:
        with open(processed_train_file, "w", encoding = "utf8") as processed: 
            for line in train:
                # If not end of tweet
                if line.strip():
                    token, tag = line.split("\t")
                    tag = tag.strip()
                    #If token is of the format "@USER..."
                    if re.match("^@USER_[a-zA-Z0-9]+('?s)?$", token):
                        # Group them all to one token "@USER"
                        new_token = "@USER"
                        processed.write(f"{new_token}\t{tag}\n")
                    #IF token is a link of the format "http://..."
                    elif re.match("http://\S+", token): 
                        # Group them all to one token "http://"
                        new_token = "http://"
                        processed.write(f"{new_token}\t{tag}\n")
                    else : 
                        processed.write(line)
                else : 
                    processed.write(line)

# Automatically generate the preprocessed train data and write it to a new file, twitter_train_proc.txt
preprocessing(f"{ddir}/twitter_train.txt", "twitter_train_proc.txt")

# Compute new output probabilities using the preprocessed train data
def compute_output2(proc_train_file, output_file): 
    # Token = word, Tag = twitter_tags.txt

    # Create a dictionary of dictionaries to store the counts of a word being associated to a specific tag
    # key: token. value: {tag: no.of times token has tag}
    token_tag_counts = {}
    # Create dictionary to store the counts of each tag for denominator
    # key: tag. value: no.of times tag appears in training text
    tag_counts = {}

    # Open the training_file and split the tokens and tag
    with open(proc_train_file, "r", encoding="utf8") as train:
        for line in train:
            # If not end of tweet
            if line.strip():
                token, tag = line.split("\t")
                tag = tag.strip()

                # If tokens / tags not inside their respective dictionaries
                if token not in token_tag_counts:
                    # if token not inside the token_tag_counts dict, add it in with count = 0
                    token_tag_counts[token] = {}

                if tag not in tag_counts:
                    # if tag is not already in the tag_counts dictionary, add it with count = 0
                    tag_counts[tag] = 0

                # Increment the count of the tag in the tag_counts dict
                tag_counts[tag] += 1

                # If the tag is not a key in the dict for token, add it with count = 0
                if tag not in token_tag_counts[token]:
                    token_tag_counts[token][tag] = 0

                # Increment the count of the tag for the token in the token_tag_counts dict
                token_tag_counts[token][tag] += 1

    # Dictionary of dictionary to store the output probabilties
    # key: token. value: {tag: P(token|tag)=count(token, tag)/count(tag)}
    output_probs = {}

    # For loop through each token in the token_tag_counts dictionary
    for token in token_tag_counts:
        # Creating a dict to store the output probabilties for this token.
        output_probs[token] = {}
        # Loop through each tag for this token in the token_tag_counts dict and use MLE
        for tag in token_tag_counts[token]:
            # numerator = number of times token w is associated with tag j
            # denominator = number of times tag j appears
            # Using smoothing constant = 1
            output_probs[token][tag] = (token_tag_counts[token][tag] + 1) / (
                tag_counts[tag] + 1 * (len(token_tag_counts) + 1)
            )

    # Writing to output file
    with open(output_file, "w", encoding="utf-8") as output:
        for token in output_probs:
            for tag in output_probs[token]:
                output.write(f"{token}\t{tag}\t{output_probs[token][tag]}\n")
                """Sample Output
                    RT	~	0.03445525602169405
                    RT	N	0.00025342118601115053
                """
# Compute new transition probabilities using the preprocessed train data
def compute_trans2(proc_train_file, tags_file, output_file):
    # initialize transition count dictionary to count the number of transitions between different pos tags.
    transition_counts = {}

    # initialize tag count dictionary to count number of times a tag appears.
    tag_counts = {}

    # Read in twitter_tags.txt
    with open(tags_file, "r", encoding="utf-8") as f:
        tags = [tag.strip() for tag in f.readlines()]
        # print(tags)

    # Open the training_file and split the tokens and tag and fill up both dictionaries.
    with open(proc_train_file, "r", encoding="utf8") as train:
        # Set previous tag to = START for first.
        previous_tag = "START"
        for line in train:
            # If not end of tweet
            if line.strip():
                token, tag = line.split("\t")
                tag = tag.strip()

                if previous_tag == "START":
                    if "START" not in tag_counts:
                        tag_counts["START"] = 0
                    tag_counts["START"] += 1

                if previous_tag != None:
                    # Set transition to a tuple of previous_tag and current tag.
                    trans = (previous_tag, tag)
                    if trans not in transition_counts:
                        # If trans does not exist in transition_counts, add to transition_counts dictionary
                        transition_counts[trans] = 0
                    transition_counts[trans] += 1

                # If tag does not exist in tag counts, add to tagcounts dictionary
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1
                # Set current tag to previous_tag to prepare for next iteration.
                previous_tag = tag

            # if end of existing tweet, set previous tag back to START.
            else:
                trans = (previous_tag, "STOP")
                if trans not in transition_counts:
                    # If trans does not exist in transition_counts, add to transition_counts dictionary
                    transition_counts[trans] = 0
                transition_counts[trans] += 1
                # If tag does not exist in tag counts, add to tagcounts dictionary
                if "STOP" not in tag_counts:
                    tag_counts["STOP"] = 0
                tag_counts["STOP"] += 1
                # Set tag back to start for next iteration.
                previous_tag = "START"

    # Compute the transition probabilities with the 2 dictionaries.
    transition_probs = {}
    smoothing_const = 10
    num_tags = len(tags)  # should = 25
    # print("number of pos tags: " + str(num_tags))
    for transition, count in transition_counts.items():
        prev_tag = transition[0]  # first element of tuple
        # smoothing transitionprobability
        prob = (count + smoothing_const) / (tag_counts[prev_tag] + smoothing_const * (num_tags + 1))
        transition_probs[transition] = prob

    # Set any transition probabilities that do not occur in the training data to smoothed value
    tags.append("START")
    tags.append("STOP")
    # print(tags)
    for tag1 in tags:
        for tag2 in tags:
            if (tag1, tag2) not in transition_probs:
                transition_probs[(tag1, tag2)] = smoothing_const / (tag_counts[tag1] + smoothing_const * (num_tags + 1))

    # write the transition probabilities generated to trans_probs.txt
    with open(output_file, "w", encoding="utf-8") as f:
        for transition, prob in transition_probs.items():
            f.write(f"{transition[0]}\t{transition[1]}\t{prob:.6f}\n")
            """Sample Output
                ~	@	0.466038
                @	~	0.242804
                ~	O	0.054717
            """

# Generating output_probs2.txt and trans_probs2.txt for viterbi2
compute_output2(f"{ddir}/twitter_train_proc.txt", "output_probs2.txt")
compute_trans2(f"{ddir}/twitter_train_proc.txt", f"{ddir}/twitter_tags.txt", "trans_probs2.txt")

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    #Remember when reading in test file to convert all @USERs to @USER also.
    #Read all tags as a list
    with open(in_tags_filename, 'r', encoding="utf-8") as in_tags:
        #tags: a list containing 25 tags
        tags = [tag.strip() for tag in in_tags.readlines()]
    
    #Read the output probabilities from output_probs2.txt
    with open(in_output_probs_filename, 'r', encoding="utf-8") as naive_output:
        #output_probs: a dictionary of dictionary
        #key:token. value:{tag:P(token|tag)}.
        output_probs = {}
        for line in naive_output: 
            token, tag, prob = line.strip().split('\t')

            if tag not in output_probs: 
                output_probs[tag] = {}
            
            output_probs[tag][token] = float(prob)

    #Read the transition probabilities from trans_probs2.txt
    with open(in_trans_probs_filename, 'r', encoding="utf-8") as trans:
        #trans_probs: a dictionary of dictionary
        #key:tag i. value:{tag j: P(i -> j|tag i)}.
        trans_probs = {}
        for line in trans: 
            curr_tag, next_tag, prob = line.strip().split('\t')

            if curr_tag not in trans_probs: 
                trans_probs[curr_tag] = {}
            
            trans_probs[curr_tag][next_tag] = float(prob)



    with open(in_test_filename, 'r', encoding="utf-8") as inputs, open(out_predictions_filename, 'w', encoding="utf-8") as outputs:
        #For loop each line in input file

        curr_tweet = []
        output_predictions = []
        for line in inputs : 
            #If not end of tweet
            if line.strip() : 
                token = line.strip()
                #Tokens in test data are also preprocessed using the same techniques
                if re.match("^@USER_[a-zA-Z0-9]+('?s)?$", token):
                    new_token = "@USER"
                    curr_tweet.append(new_token)
                elif re.match("http://\S+", token): 
                    new_token = "http://"
                    curr_tweet.append(new_token)
                else : 
                    curr_tweet.append(token)
            else:
                curr_tweet_predictions = viterbi_tweet(curr_tweet, tags, output_probs, trans_probs)
                output_predictions += curr_tweet_predictions
                output_predictions.append(" ")
                curr_tweet = []

        for predict_tag in output_predictions:
            #Write to output file
            outputs.write(f"{predict_tag}\n")
    


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth:
            correct += 1
    return correct, len(predicted_tags), correct / len(predicted_tags)


def run():
    """
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    """
    # Working dir
    ddir = (
        '/Users/yanghaoying/Desktop/Y2S2/BT3102/Project/project-files'
    )

    in_train_filename = f"{ddir}/twitter_train.txt"

    # naive_output_probs_filename = f"{ddir}/naive_output_probs.txt"

    in_test_filename = f"{ddir}/twitter_dev_no_tag.txt"
    in_ans_filename = f"{ddir}/twitter_dev_ans.txt"
    # naive_prediction_filename = f"{ddir}/naive_predictions.txt"
    # naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    # correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    # print(f"Naive prediction accuracy:     {correct}/{total} = {acc}")

    # naive_prediction_filename2 = f"{ddir}/naive_predictions2.txt"
    # naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2,)
    # correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    # print(f"Naive prediction2 accuracy:    {correct}/{total} = {acc}")

    trans_probs_filename = f"{ddir}/trans_probs.txt"
    output_probs_filename = f"{ddir}/output_probs.txt"

    in_tags_filename = f"{ddir}/twitter_tags.txt"
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                     viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                      viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')


if __name__ == "__main__":
    run()
