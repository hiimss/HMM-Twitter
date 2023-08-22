'''
BT3102 Group Fifteen
Lee Shan Shan	A0241238L
Zhang Zishuo	A0239837M
Yang Haoying	A0238149X
Lim Wei Xuan	A0233827B

Naive prediction accuracy:     1071/1378 = 0.7772133526850508
Naive prediction2 accuracy:    1072/1378 = 0.7779390420899854
'''


#HELPER FUNCTION
import re

def MLE(train_file) :
    #Token = word, Tag = twitter_tags.txt

    #Create a dictionary of dictionaries to store the counts of a word being associated to a specific tag
    #key: token. value: {tag: no.of times token has tag}
    token_tag_counts = {}
    # Create dictionary to store the counts of each tag for denominator
    #key: tag. value: no.of times tag appears in training text
    tag_counts = {}


    #Open the training_file and split the tokens and tag
    with open(train_file, 'r',  encoding="utf8") as train: 
        for line in train : 
            #If not end of tweet
            if line.strip() :
                token, tag = line.split('\t')
                 
                # If tokens / tags not inside their respective dictionaries 
                if token not in token_tag_counts: 
                    #if token not inside the token_tag_counts dict, add it in with count = 0
                    token_tag_counts[token] = {}

                if tag not in tag_counts : 
                    #if tag is not already in the tag_counts dictionary, add it with count = 0
                    tag_counts[tag] = 0

                #Increment the count of the tag in the tag_counts dict
                tag_counts[tag] += 1

                #If the tag is not a key in the dict for token, add it with count = 0
                if tag not in token_tag_counts[token] : 
                    token_tag_counts[token][tag] = 0

                #Increment the count of the tag for the token in the token_tag_counts dict
                token_tag_counts[token][tag] += 1

    #Dictionary of dictionary to store the output probabilties
    #key: token. value: {tag: P(token|tag)=count(token, tag)/count(tag)}
    output_probs = {}

    #For loop through each token in the token_tag_counts dictionary
    for token in token_tag_counts : 
        #Creating a dict to store the output probabilties for this token.
        output_probs[token] = {}
        #Loop through each tag for this token in the token_tag_counts dict and use MLE
        for tag in token_tag_counts[token] : 
            #numerator = number of times token w is associated with tag j
            #denominator = number of times tag j appears
            #Using smoothing constant = 1
            output_probs[token][tag] = (token_tag_counts[token][tag] + 1)/ ( tag_counts[tag] + 1 * (len(token_tag_counts) + 1) )

    #Writing to output file
    with open('naive_output_probs.txt', 'w', encoding="utf-8") as output: 
        for token in output_probs: 
            for tag in output_probs[token]: 
                output.write(f"{token}\t{tag}\t{output_probs[token][tag]}\n")
                '''Sample Output
                RT	~
	                0.03445525602169405
                '''
#Get naive_output_probs.txt
ddir = 'C:/Users/wxlim/OneDrive/Desktop/Y2S2/BT3102/Project/project-files/projectfiles'
MLE(f'{ddir}/twitter_train.txt')

# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    '''
    in_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    
    out_prediction_filename = f'{ddir}/naive_predictions.txt'
    '''

    #Read the output probabilities from naive_output_probs.txt
    with open(in_output_probs_filename, 'r', encoding="utf-8") as naive_output:
        #output_probs: a dictionary of dictionary
        #key:token. value:{tag:P(token|tag)}.
        output_probs = {}
        for line1, line2 in zip(naive_output,naive_output) : 
            token, tag = line1.strip().split('\t')
            prob = line2.strip()

            if token not in output_probs: 
                output_probs[token] = {}
            
            output_probs[token][tag] = float(prob)
    
    with open(in_test_filename, 'r', encoding="utf-8") as inputs, open(out_prediction_filename, 'w', encoding="utf-8") as outputs:
        #For loop each line in input file
        for line in inputs : 
            #If not end of tweet
            if line.strip() : 
                tokens = line.strip().split()

                #Initialize the predicted tag for a word
                predict_tag = ""

                #For each token, assign their predicted tag
                for token in tokens: 
                    #If token is unknown, not in our training set, assume Noun 
                    if token not in output_probs : 
                        #Tag @ to all USER tokens
                        if re.match('^@USER_[0-9a-f]+$', token):
                            predict_tag = "@"
                        else : 
                            predict_tag = "N"

                    else : 
                        #Get the output prob for the token and choose the tag with highest prob
                        probs = output_probs[token]
                        predict_tag = max(probs, key = probs.get)
                
                    #Write to output file
                    outputs.write(f"{predict_tag}\n")
                    # Naive prediction accuracy:     1071/1378 = 0.7772133526850508




def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    '''
    in_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_train_filename = f'{ddir}twitter.train.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    
    out_prediction_filename = f'{ddir}/naive_predictions.txt'

    3a) A better approach is to estimate j* using j* = argmaxj [P(y = j|x = w)]
        We want to calculate and maximize the probability of observing a given token w for each possible tag j
        To do so, we can use the bayes theorem : 
        P(y = j | x = w) = [ P(x = w | y = j) * P(y = j) ] / P(x = w)

        P(x = w | y = j) is the probability of observing token w given that the tag is j, 
            -  count of the number of times token w appears with tag j, 
            divided by the count of the number of times tag j appears in the training data

        P(y = j) is the prior probability of tag j, 
            -  count of the number of times tag j appears in the training data, 
            divided by the total number of tags in the training data.

        P(x = w) is the probability of observing token w regardless of the tag
            -  count of the number of times token w appears in the training data, 
            divided by the total number of tokens in the training data

        Using Maximum Likelihood estimation, we get estimate these probabilities.
        P(y = j) = count(j) / count(total tags)
        P(x = w) = count(w) / count(total tokens)
        p(x = w | y = j) = count(w,j) / count(j) can get from our naive_output_probs file from MLE Function in 2a.

        Hence we can simplify [ P(x = w | y = j) * P(y = j) ] / P(x = w) to : 
        [p(x = w | y = j) * count(j) / count(total tags) ] / [count(w) / count(total tokens)]
        
        Since each token is assigned to a single tag, hence count(total tags) == count(total tokens) and we can simplify the formula even further to : 
        [ p(x = w | y = j) ] * [ count(j) / count(w) ]

        Thus we want to maximize this probability.

    '''

    #Read the output probabilities from naive_output_probs.txt
    with open(in_output_probs_filename, 'r', encoding="utf-8") as naive_output:
        #output_probs: a dictionary of dictionary
        #key:token. value:{tag:P(token|tag)}.
        output_probs = {}
        for line1, line2 in zip(naive_output,naive_output) : 
            token, tag = line1.strip().split('\t')
            prob = line2.strip()

            if token not in output_probs: 
                output_probs[token] = {}

            output_probs[token][tag] = float(prob)
        
    #Read the input training file from twitter_train.txt
    with open(in_train_filename, 'r',  encoding="utf8") as train:
        #Dictionary to count tokens:
        #key: token. value: no. of times token appear in train text
        simple_token_counts = {}

        #Dictionary to count tags:
        #key tag. value: no. of times tag appear in train text
        simple_tag_counts = {}
        
        for line in train : 
            #If not end of tweet
            if line.strip() :
                token, tag = line.strip().split('\t')
                # If tokens / tags not inside their respective dictionaries 
                if token not in simple_token_counts: 
                    #if token not inside the simple_token_counts dict, add it in with count = 0
                    simple_token_counts[token] = 0

                if tag not in simple_tag_counts : 
                    #if tag is not already in the simple_tag_counts dictionary, add it with count = 0
                    simple_tag_counts[tag] = 0

                #Increment the count of the token in the simple_token_counts dict
                simple_token_counts[token] += 1

                #Increment the count of the tag in the simple_tag_counts dict
                simple_tag_counts[tag] += 1
         
    with open(in_test_filename, 'r', encoding="utf-8") as inputs, open(out_prediction_filename, 'w', encoding="utf-8") as outputs:
        #For loop each line in input file
        for line in inputs : 
            #If not end of tweet
            if line.strip() : 
                tokens = line.strip().split()

                #Initialize the predicted tag for a word
                predict_tag = ""

                #For each token, assign their predicted tag
                for token in tokens: 
                    #If token is unknown, not in our training set, assume Noun
                    if token not in output_probs : 
                        #Tag @ to all USER tokens
                        if re.match('^@USER_[0-9a-f]+$', token):
                            predict_tag = "@"
                        else : 
                            predict_tag = "N"

                    else : 
                        #Get the output prob for the token and choose the tag with highest prob
                        #probs: a dictionary. key: tag. value: P(token|tag)
                        probs = output_probs[token]

                        #lambda function through probs to get tag with highest P(tag|token)
                        #P(tag|token) = P(token|tag) * ( count(tag) / (count(token) )
                        predict_tag = max(probs, key = lambda k : (probs[k] * (simple_tag_counts[k] / simple_token_counts[token])))
                
                    #Write to output file
                    outputs.write(f"{predict_tag}\n")
                    #Naive prediction2 accuracy:    1072/1378 = 0.7779390420899854

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''
    # Working dir
    ddir = 'C:/Users/wxlim/OneDrive/Desktop/Y2S2/BT3102/Project/project-files/projectfiles'

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    # trans_probs_filename =  f'{ddir}/trans_probs.txt'
    # output_probs_filename = f'{ddir}/output_probs.txt'

    # in_tags_filename = f'{ddir}/twitter_tags.txt'
    # viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    # viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
    #                 viterbi_predictions_filename)
    # correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    # print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    # trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    # output_probs_filename2 = f'{ddir}/output_probs2.txt'

    # viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
    #                  viterbi_predictions_filename2)
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()
