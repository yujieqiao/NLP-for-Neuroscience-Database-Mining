import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("f1_specter.json") as peach:
    f1_specter = json.load(peach)

with open("f1_linkbert_low.json") as kiwi:
    f1_linkbert = json.load(kiwi)


# ## plot figure 1 
# # Create a dataframe
# metadata=[]
# specter=[]
# linkbert=[]
# for k,v in f1_specter.items():
#     for k2,v2 in f1_linkbert.items():
#         if k==k2:
#             # print(k)
#             metadata.append(k[:19])
#             specter.append(v)
#             linkbert.append(v2)
# print(len(metadata))    
# # print(len(specter))  
# # print(len(linkbert)) 

# df = pd.DataFrame({'Metadata':metadata, 'f1_specter':specter, 'f1_linkbert':linkbert})
# # print(df)


# # Reorder it following the values of the first value:
# ordered_df = df.sort_values(by='f1_specter')
# my_range=range(1,len(df.index)+1)
 
# # The horizontal plot is made using the hline function
# plt.hlines(y=my_range, xmin=ordered_df['f1_specter'], xmax=ordered_df['f1_linkbert'], color='grey', alpha=0.4)
# plt.scatter(ordered_df['f1_specter'], my_range, color='skyblue', alpha=1, label='F1_SPECTER')
# plt.scatter(ordered_df['f1_linkbert'], my_range, color='green', alpha=0.4 , label='F1_LinkBert')
# plt.legend()
 
# # Add title and axis names
# plt.yticks(my_range, ordered_df['Metadata'],fontsize=6)
# plt.title("Comparison of the F1 score of SPECTER and the F1 score of LinkBert", loc='left')
# plt.xlabel('F1 score')
# plt.ylabel('Metadata')

# # Show the graph
# plt.show()


## plot top 20 tags

metadata=[]
specter=[]
for k,v in f1_specter.items():
    metadata.append(k[:19])
    specter.append(v)
print(len(metadata))    

df = pd.DataFrame({'Metadata':metadata, 'f1_specter':specter})
# print(df)


top20 = df.sort_values('f1_specter', ascending=False)[:25]



# Creating a case-specific function to avoid code repetition
def plot_hor_vs_vert(x, y, xlabel, ylabel, rotation,
                     tick_bottom, tick_left):
    sns.barplot(x, y, data=top20, color='slateblue',ci=None)
    plt.title('Top 20 metadata tags prediction using SPECTER', fontsize=17)
    plt.xlabel(xlabel, fontsize=10)
    plt.xticks(fontsize=8, rotation=rotation)
    plt.ylabel(ylabel, fontsize=8)
    plt.yticks(fontsize=8)
    sns.despine(bottom=False, left=False)
    plt.tick_params(bottom=tick_bottom, left=tick_left)

    return None

# plot_hor_vs_vert(x='Metadata', y='f1_specter',
#                  xlabel=None, ylabel='Servings per sperson',
#                  rotation=90, tick_bottom=False, tick_left=True)
plot_hor_vs_vert(x='f1_specter', y='Metadata',
                 xlabel='F1 score', ylabel=None,
                 rotation=None, tick_bottom=False, tick_left=False)
plt.show()