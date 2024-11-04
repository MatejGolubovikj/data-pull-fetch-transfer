#################################
#Import 
#################################

import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit.algorithms import Recommender, als, user_knn, item_knn, svd, user_knn
from lenskit.algorithms.ranking import TopN
from lenskit.algorithms.basic import UnratedItemCandidateSelector, Popular
from lenskit import crossfold as cf

from collections import deque
import random
import gc

pd.set_option('mode.chained_assignment', None)


#################################
#Load Ratings with split
#################################

ratings=pd.read_parquet("Data\Experiment_Data\Ratings_with_Split.parquet")

#################################
#Testing storages
#################################
print("Testing storage")
ratings.head(5).to_parquet("Data/29102024_Full_Experiment_Data_SongSpecificAge/TEST.parquet")
print("Able to access storage")


#################################
#Load User Age Data
#################################

user_full=pd.read_csv("Data/Users/users.tsv", sep='\t')
user_full=user_full[['user_id','age']]
user_full.columns=['user','age']

#################################
#Select users in experiment and create a user-> age dictonary (result: a user, age datafareme and a user-> age dictonary)
#################################

user_age=user_full[user_full['user'].isin(ratings.user.unique())]
user_age_index=dict(zip(list(user_age['user']), list(user_age['age'])))


#################################
#Load Song data
#################################

song_info=pd.read_parquet("Data/Tracks/track_spotify_extra_info_All.parquet")
song_info['track_id']=song_info['track_id'].astype(int)

#################################
#Extract song release year from song data (To calculate SSA-Song Specific Age )
#################################
song_year_info=song_info[['track_id','release_year']]
song_year_info.columns=['item','release_year']



################################
# TO DO
# Adjust SSA such that if desired distribution among categories cannot be reached, ditribute the requred songs among othe cats in proper proportions (Alan's note)
#################



#################################
# Define a function whcih converts song specific age to the categories of:
# Young (Y, SSA of 0 to 21 years), 
# Young Adulthood (YgA, SSA of 21 to 40 years), 
# Middle Adulthood (MdA, SSA of >40 years i.e. 40 to 61) and,
# Before Time (BT, SSA <0, Songs released before the user was born)
#################################
def fileter_SSA(SSA):
    if SSA>=0 and SSA<21:
        return "Y"
    elif SSA>20 and SSA<41:
        return "YgA"
    elif SSA>40:
        return 'MdA'
    else: #The age is less than 0
        return 'BT'

#################################
# Define a function whcih for a given user selects a given quantity of songs per SSA Category
# Quatity is defined per age group
#################################

def adjust_user_per_SSA(recs,SSA_quants,N=10):
   #Save original Top N
   topn=recs.head(N)

   #Add SSA and divide per SSA category (NO MORE)
   #recs= recs.merge(ssa,on=['user','item'],how='left')

   #Find number of quants Requred per SSA time
   BT_n=SSA_quants[0]
   Y_n=SSA_quants[1]
   YgA_n=SSA_quants[2]
   MdA_n=SSA_quants[3]

   #Select the top rated songs per SSA type and compose a recomended list
   recs_BT=recs[recs['SSA_tag']=="BT"].sort_values('score', ascending=False).head(BT_n)
   recs_Y=recs[recs['SSA_tag']=="Y"].sort_values('score', ascending=False).head(Y_n)
   recs_YgA=recs[recs['SSA_tag']=="YgA"].sort_values('score', ascending=False).head(YgA_n)
   rec_MdA=recs[recs['SSA_tag']=="MdA"].sort_values('score', ascending=False).head(MdA_n)

   recs=pd.concat([recs_BT,recs_Y,recs_YgA,rec_MdA])

   #If we could not find enough songs per SSA category to fill 10 recs add other top rated songs to fill list
   if len(recs)<N:
      
      leftover_n=N-len(recs)
      leftover_additions=topn[~topn['item'].isin(recs['item'])].sort_values('score', ascending=False).head(leftover_n)
      leftover_additions['SSA_tag']=['ExtNA']*len(leftover_additions)
      recs=pd.concat([recs,leftover_additions])

   recs=recs.sort_values('score', ascending=False)


   #Recalculate rank
   rec_len=min(N,len(recs))
   recs['rank']=list(range(1,rec_len+1))

   return recs

#################################
# Define a function whcih for a adjust per SSA cat for all users
#################################

def adjust_per_SSA(recs,SSA_quants,song_year_info,user_age,N=10):
    
    #Calculate SSA for all user-song pair in the recommended items
    recs=recs.merge(song_year_info,on=['item'],how='left')
    recs=recs.merge(user_age,on=['user'],how='left')
    recs['birth_year']=2013-recs['age']
    recs['SSA']=recs['release_year']-recs['birth_year']

    #Clasify SSA into SSA categories
    recs['SSA_tag']=[fileter_SSA(SSA) for SSA in recs['SSA'] ]

   
    # For all users
    final_list=None
    set_list=True
    users=set(recs['user'])

    for user in users:
        
        # Adjust recomendations such that they follow the correct distribution
        user_list=recs[recs['user']==user]
        user_list=user_list.reset_index(drop=True)
        
        user_adujusted_list=adjust_user_per_SSA(user_list,SSA_quants,N=N)

        # Add all recoomedantions to a list
        if set_list:
            final_list=user_adujusted_list
            set_list=False
        else:
            final_list=pd.concat([final_list,user_adujusted_list], sort=False)

    #retrun list of recomendations for all users
    #SSA_s=final_list['SSA_tag']    
    return final_list.drop(['SSA','age','release_year','birth_year','SSA_tag'],axis=1)


#Helper function (converst a series to a dataframe)
def convert_to_DF(series):
    return pd.DataFrame(series).T

###########
#Define a function which evalutes recomendations in terms of:
#NDCG@k, Percision@k, Recall@k, Hit Rate@k
###########

def evaluate_recomendations(recomendations, truth, k=100):
    
    analysis = topn.RecListAnalysis()
    analysis.add_metric(topn.ndcg,k=k)
    analysis.add_metric(topn.precision,k=k)
    analysis.add_metric(topn.recall,k=k)
    analysis.add_metric(topn.hit)
    results = analysis.compute(recomendations, truth)
    
    return results



###########
#Define a function which evalutes diversity in recomendations based on shanon enthropy
###########
def shanon_enthropy(rec_list):
    
    rec_list=rec_list.groupby('item').count().reset_index()
    rec_list=list(rec_list['user'])
    
    total_recs=sum(rec_list)
    probs=np.array(rec_list)/total_recs
    
    enthropy=sum(probs*np.log2(probs))*-1
    
    return enthropy


####################
#Define age ranges and neighbours to test
####################
age_ranges=[(10,20),(16,26),(26,36),(36,46),(46,56),(49,59),(49,55),(55,61),(10,64)] 
#[(10,64),(10,20),(16,26),(26,36),(36,46),(46,56),(49,59),(49,55),(55,61)] 
#[(10,64),(10,61),(10,20),(21,30),(31,40),(41,50),(51,61),(10,15),(16,25),(26,35),(36,45),(46,55),(49,59),(49,55),(56,61)]
neighbourhoods=[6,8,12,18,24,36,50,60,70,100,110,120,150]
#[36,50,50,36,36,24,24,18,12]"Recs_per_Age_Group OLD"

####################
#We will generate 100 recs and adjust them to 10 recs
####################
n=100 #200
N=10

set_results=True

####################
#Define SSA distribution per age
####################
SSA_quantaty_per_age=[
[1,6,3,0], #(10, 64)
[2,8,0,0], # (10, 20)
[1,7,2,0], # (16, 26)
[1,4,5,0], # (26, 36)
[1,2,6,1], # (36, 46)
[0,2,3,5], # (46, 56)
[0,2,3,5], # (49, 59)
[0,2,3,5], # (49, 55)
[0,2,2,6] # (55, 61)
]

user_age_index_df=user_age


####################
#Test Configurations
####################

#per CV split
for cv_iter in range(1,5+1):
    
    #Select train test split
    test=ratings[ratings['split']==cv_iter]
    train=ratings.drop(test.index)
    
    test=test.drop('split', axis=1)
    train=train.drop('split', axis=1)
    
    print("Testing in CV iteration:",cv_iter)
    
    ###For each neighbourhood size ####
    for neighbours in neighbourhoods:

        ###Train a user KNN model with exact number of neighbours k ####
        print("Training")
        predictor = user_knn.UserUser(neighbours,min_nbrs=neighbours,center=False,feedback='implicit',use_ratings=False)
        Unseen_item_selector = UnratedItemCandidateSelector()
        recommender = TopN(predictor, Unseen_item_selector)    
        predictor.fit(train)
        Unseen_item_selector.fit(train)
        
        ### Generate recommendations for all users using the model ####
        print("Generating Recomendations")
        recomendations_all=batch.recommend(recommender, train.user.unique(), n ,n_jobs=15)
        
        ### Test both with adjusting for SSA and no adjusting in the recommendations ####
        for do_adjust in ["Yes",'No']:
            
            print("Testing with adjustment:",do_adjust)
                
            ### For all age groups ####
            for i in range(len(age_ranges)):

                ####F ind the age range and desirted SSA distribution of the age group ####
                age_range_start,age_range_end=age_ranges[i]
                #neighbours=neighbourhoods[i]
                SSA_distribution=SSA_quantaty_per_age[i]


                age_range_str=str(age_range_start)+'-'+str(age_range_end)
                
                #### Select users in the age group ####
                users_in_age_range=user_age_index_df[(user_age_index_df['age']>=(age_range_start)) & 
                                        (user_age_index_df['age']<=(age_range_end))]                

                info= "age range "+age_range_str+" with "+str(neighbours)+" neighbours and " + do_adjust + " SSA adjustment" 
                print('Processing',info,"in itteration",cv_iter)

                #### Find the test set for the age group ####
                truth_in_age_range=test[test['user'].isin(users_in_age_range.user.unique())]
                
                #### Select the generated recomendations per age group ####
                recs_in_age_range = recomendations_all[recomendations_all['user'].isin(users_in_age_range.user.unique())]
                #recomendations_all.merge(users_in_age_range[['user']], how='inner', on='user')

                print("============")
                print("WE ARE RECCOEMEDNING TO")
                print("============")
                
                testing=recs_in_age_range.groupby('user').count().reset_index()
                print(len(testing[testing['score']>0]))
                
                print("USERS")
                print("============")

                
                
                #### if we are adjusting per SSA adjust otherwise select top 10 ####
                if do_adjust=='Yes':
                    print("Adjusting Recs per SSA")
                    recs_in_age_range =adjust_per_SSA(recs_in_age_range,SSA_distribution,song_year_info,user_age,N=N)
                else:
                    recs_in_age_range=recs_in_age_range.sort_values('rank',ascending=True).groupby('user').head(N).reset_index()


                #### Evaluate recomendations per user ####
                results_i=evaluate_recomendations(recs_in_age_range, truth_in_age_range, k=N)
                
                
                #filename='User_Results_'+age_range_str+'_'+str(neighbours)+'_neig_'+do_adjust+"_adj_CViter_"+str(cv_iter)+'.csv'
                
                #results_i=results_i.reset_index()
                #results_i.to_csv('Data/Experiment_Data/CV_iters/'+filename, index=False)
                
                #### Calcualte avg result ####
                results_i=results_i[["ndcg","precision","recall","hit"]].mean()
                results_i=convert_to_DF(results_i)
                
                #### Calcualte shannon diversity ####
                diversity_pressent=shanon_enthropy(recs_in_age_range)

                print('Final Evaluation for recomendations in', info, 'and',diversity_pressent,"measured Shannon diversity")
                print(results_i)


                #### Add configuration data to results ####
                #results_i['List_Len']=[k]*len(results_i)
                results_i['Neighbours']=[neighbours]*len(results_i)
                #results_i['Diversity_adjustment']=[diversity]*len(results_i)
                #results_i['Intralist_Diversity_calculated']=[Intralist_diversity_avg]*len(results_i)
                results_i['Shannon_diversity_pressent']=[diversity_pressent]*len(results_i)
                results_i['Age_range']=[age_range_str]*len(results_i)
                results_i['SSA_Adjustment']=[do_adjust]*len(results_i)
                results_i['CV_iter']=[cv_iter]*len(results_i)
                
                #### Add this iteration to final reusults ####
                if set_results:
                    results=results_i
                    set_results=False
                else:
                    results=pd.concat([results,results_i], sort=False)
    #### Save results per fold ####
    sifix="_at_fold_"+str(cv_iter)
    results.to_parquet("Data/29102024_Full_Experiment_Data_SongSpecificAge/Results_Cross_validationNew_5_10_2"+sifix+".parquet")
#### Save final results ###
results.to_parquet("Data/29102024_Full_Experiment_Data_SongSpecificAge/Results_Cross_validationNew_5_10_2_Final.parquet")    
    
#### Preview Results ###    
results.head()                
