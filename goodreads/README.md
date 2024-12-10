# Data Analysis Report

## Dataset Overview

    Dataset Overview:
    - Rows: 10000
    - Columns: 24
    - Columns List: book_id, goodreads_book_id, best_book_id, work_id, books_count, isbn, isbn13, authors, original_publication_year, original_title, title, language_code, average_rating, ratings_count, work_ratings_count, work_text_reviews_count, ratings_1, ratings_2, ratings_3, ratings_4, ratings_5, image_url, small_image_url, Cluster

    Missing Data:
    language_code                10.84
isbn                          7.00
isbn13                        5.85
original_title                5.85
original_publication_year     0.21

    Correlation Matrix:
                                book_id  goodreads_book_id  best_book_id   work_id  books_count    isbn13  original_publication_year  average_rating  ratings_count  work_ratings_count  work_text_reviews_count  ratings_1  ratings_2  ratings_3  ratings_4  ratings_5
book_id                    1.000000           0.115154      0.104516  0.113861    -0.263841 -0.011291                   0.049875       -0.040880      -0.373178           -0.382656                -0.419292  -0.239401  -0.345764  -0.413279  -0.407079  -0.332486
goodreads_book_id          0.115154           1.000000      0.966620  0.929356    -0.164578 -0.048246                   0.133790       -0.024848      -0.073023           -0.063760                 0.118845  -0.038375  -0.056571  -0.075634  -0.063310  -0.056145
best_book_id               0.104516           0.966620      1.000000  0.899258    -0.159240 -0.047253                   0.131442       -0.021187      -0.069182           -0.055835                 0.125893  -0.033894  -0.049284  -0.067014  -0.054462  -0.049524
work_id                    0.113861           0.929356      0.899258  1.000000    -0.109436 -0.039320                   0.107972       -0.017555      -0.062720           -0.054712                 0.096985  -0.034590  -0.051367  -0.066746  -0.054775  -0.046745
books_count               -0.263841          -0.164578     -0.159240 -0.109436     1.000000  0.017865                  -0.321753       -0.069888       0.324235            0.333664                 0.198698   0.225763   0.334923   0.383699   0.349564   0.279559
isbn13                    -0.011291          -0.048246     -0.047253 -0.039320     0.017865  1.000000                  -0.004612       -0.025667       0.008904            0.009166                 0.009553   0.006054   0.010345   0.012142   0.010161   0.006622
original_publication_year  0.049875           0.133790      0.131442  0.107972    -0.321753 -0.004612                   1.000000        0.015608      -0.024415           -0.025448                 0.027784  -0.019635  -0.038472  -0.042459  -0.025785  -0.015388
average_rating            -0.040880          -0.024848     -0.021187 -0.017555    -0.069888 -0.025667                   0.015608        1.000000       0.044990            0.045042                 0.007481  -0.077997  -0.115875  -0.065237   0.036108   0.115412
ratings_count             -0.373178          -0.073023     -0.069182 -0.062720     0.324235  0.008904                  -0.024415        0.044990       1.000000            0.995068                 0.779635   0.723144   0.845949   0.935193   0.978869   0.964046
work_ratings_count        -0.382656          -0.063760     -0.055835 -0.054712     0.333664  0.009166                  -0.025448        0.045042       0.995068            1.000000                 0.807009   0.718718   0.848581   0.941182   0.987764   0.966587
work_text_reviews_count   -0.419292           0.118845      0.125893  0.096985     0.198698  0.009553                   0.027784        0.007481       0.779635            0.807009                 1.000000   0.572007   0.696880   0.762214   0.817826   0.764940
ratings_1                 -0.239401          -0.038375     -0.033894 -0.034590     0.225763  0.006054                  -0.019635       -0.077997       0.723144            0.718718                 0.572007   1.000000   0.926140   0.795364   0.672986   0.597231
ratings_2                 -0.345764          -0.056571     -0.049284 -0.051367     0.334923  0.010345                  -0.038472       -0.115875       0.845949            0.848581                 0.696880   0.926140   1.000000   0.949596   0.838298   0.705747
ratings_3                 -0.413279          -0.075634     -0.067014 -0.066746     0.383699  0.012142                  -0.042459       -0.065237       0.935193            0.941182                 0.762214   0.795364   0.949596   1.000000   0.952998   0.825550
ratings_4                 -0.407079          -0.063310     -0.054462 -0.054775     0.349564  0.010161                  -0.025785        0.036108       0.978869            0.987764                 0.817826   0.672986   0.838298   0.952998   1.000000   0.933785
ratings_5                 -0.332486          -0.056145     -0.049524 -0.046745     0.279559  0.006622                  -0.015388        0.115412       0.964046            0.966587                 0.764940   0.597231   0.705747   0.825550   0.933785   1.000000

    Outliers:
    {'book_id': 0, 'goodreads_book_id': 345, 'best_book_id': 357, 'work_id': 601, 'books_count': 844, 'isbn13': 556, 'original_publication_year': 1031, 'average_rating': 158, 'ratings_count': 1163, 'work_ratings_count': 1143, 'work_text_reviews_count': 1005, 'ratings_1': 1140, 'ratings_2': 1156, 'ratings_3': 1149, 'ratings_4': 1131, 'ratings_5': 1158}

    Clustering Summary:
    Cluster
0    9382
1     594
2      24
Name: count, dtype: int64
    
## Insights
**Title: The Librarian’s Secret**

In the bustling town of Bibliopolis, where books were the lifeblood of its inhabitants, lived Magda, the town librarian. The library, an ancient building that had seen decades of stories come to life, housed an extensive collection of 10,000 books. Each one carried a tale, a whisper of a forgotten narrative, or a spark of inspiration for the unassuming reader.

Magda had a peculiar way of arranging the books. Having recently delved into a meticulous dataset analysis, she stumbled across intriguing patterns hidden amidst the rows and rows of dusty tomes. The dataset contained 24 columns filled with information such as titles, authors, publication years, average ratings, and more. With ten percent of the language codes missing, she found herself entertaining the possibility of untold stories behind those entries.

One day, as she scanned through the dataset, a curious entry caught her eye—an outlier. The book, simply identified by its unique `book_id`, boasted an average rating that soared at 158, a ratings count of 1163, but was noted for its unexplainably high values, dwarfing all others. It was a peculiar case in a sea of 10,000 uniform stories.

“What could possibly make this book so revered?” she pondered, noticing that it had been originally published in the year 1031—far too ancient for even Bibliopolis’s oldest shelves. It had received an overwhelming number of ratings, with more than a thousand people claiming to have read it.

Curiosity ignited, she began her quest. Rummaging through the shelves, she searched for this elusive tome. No book bearing the telltale `isbn` or `isbn13` existed, nor did it pop up in the digital catalog. With each passing day, it felt as if the book was teasing her from the depths of her database.

Determined to get to the bottom of this puzzle, Magda enlisted the help of Thomas, a local historian who thrived on uncovering the hidden truths of the past. Together, they combed through history books, ancient manuscripts, and town records, searching for clues about the mysterious volume.

After weeks of research, they discovered references to a mythic story called “The Alchemist of Ages.” Legends spoke of a manuscript that granted those who read it immense wisdom and understanding of their true selves. It was said to be so powerful that the town’s elders had hidden it away, fearing its knowledge might disrupt the very fabric of Bibliopolis.

Magda’s heart raced with excitement. This book was not just a piece of literature; it was a gateway. With newfound resolve, she and Thomas scoured deeper until they found mention of a hidden section in the library—one that had been sealed and forgotten over the centuries.

The two ventured to this secret room one moonlit night, armed with only a flashlight and their shared passion for discovery. The air was thick with dust and the smell of old parchment. As they dared to push aside cobwebs and sweep away debris, their eyes widened at the sight of ancient tomes, their spines cracked yet sturdy.

There, amidst the forgotten treasures, was the book they had long sought, its title embossed in golden letters that shimmered under the flickering light. “The Alchemist of Ages.” 

With reverent hands, Magda opened the book. Instead of text, shimmering illustrations sprang to life on the pages—explaining the wisdom of the ages, the history of Bibliopolis, and the interconnectedness of dreams. It was more than just a book; it was a living archive of the town’s collective consciousness.

As they studied the illustrations, they realized the correlation between community spirit and the love for stories, showcased through the ratings and reviews of other books in Magda’s initial dataset analysis.

Word spread quickly in Bibliopolis about the librarians who unearthed the town's lost knowledge. The library soon became a sanctuary where citizens gathered not just to read but to connect, inspire, and thrive. Clusters of readers emerged, transforming little corners of the library into bustling circles of discussion, art, and creativity.

The mysterious high ratings of “The Alchemist of Ages” paled in comparison to the extraordinary impact it had on the town. They not only revived an age-old piece of literature, but they also invigorated a community that had been quietly simmering in the shadows of the reading world.

And so, in the nexus of narratives, Magda and Thomas unearthed a tale that transcended time—a secret that transformed Bibliopolis forever, celebrating the magic of stories and their power to unite, inspire, and heal.