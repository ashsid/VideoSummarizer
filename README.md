# VideoSummarizer

Setup :
Clone the Repository\n
Create a datadir folder in this repository 
Copy paste the video for which you need the Summarised PDF in the datadir
Run "pip3 install -r requirements.txt" at the head of this repository , not datadir
Once that is done Run - "python3 finalCode.py <VideoTile> <options>" at the head of this repository , not datadir
  Video title is the title of the video that you just uploaded to the datadir
  Options is either '1' or '2' : '2' when PPT's in the video and '1' when Lecture handwritten notes on Green Board.
Once this finishes running you should be able to find a summary.pdf in the datadir along with all the other artefacts that will help you with the video.


I.INTRODUCTION

The main intention of this is to help students and working professionals in taking up online courses to have a reference handout which they can use to revise before the examination. With a growing trend in enrolment for courses online, be it e- learning sites like Udemy, Coursera, Udacity etc. or recorded classroom videos available through college platforms, a need for a summarised handbook always arises. Our main intention through the project is to provide this summarised handbook to help them better prepare and leverage it to the fullest. We have taken a modular approach to address the given problem. Mod- ules include - Transcriptor, Summariser, Segmenter, Ranking Algorithm, Condenser, Chalkboard Representation Extractor and PDF Generator.

A. Transcriptor

The task of the transcriptor part of the video summarization is to convert the video file into text format. This process of creating a transcript is performed in three steps - Firstly, the video file is converted into an audio file. Secondly, the audio file is then uploaded into Google cloud bucket. Lastly, this audio file is converted into text creating the transcript. The video file is converted in .wav format using ffmpeg, a cross-platform software that helps in recording, converting and streaming audio and video files. The video is converted into a .wav format because it has more audio clarity, which in turn helps in increasing the accuracy of the final outcome affecting the efficiency of the transcriptor as a whole. While using .wav files two parameters need to be changed as it differs for .wav format compared to other audio formats. The ‘hertz’ parameter needs to be changed to 44,100 Hz as the standard value is 16,000. Also .wav files are multi channeled, hence the ‘channels’ parameter must be changed to 2 as the default number of channels is 1. The shell property in the sub process should be set to be true as well. This .wav file is uploaded into Google cloud bucket. Further this .wav audio file is converted
into a transcript with the help of Google speech to text API. The Google speech to text API uses neural network models to perform the conversion of audio into text format. This is the process of obtaining a video converted into a transcript.

B. Summariser

We have followed the general approach for extracting a summary from the transcript using NLP based techniques. We need to follow these steps to successfully extract summary - First, we need to convert the paragraph into a sentence by splitting the paragraph at full stops. After converting the paragraph into sentences, we need to remove all the special characters, stop words and numbers from all the sentences. This is followed by tokenising all the sentences to get all the words that exist in the sentences. Next we need to find the weighted frequency of occurrences of all the words. We can find the weighted frequency of each word by dividing its frequency by the frequency of the most occurring word. Next we plug the weighted frequency in place of the corresponding words in original sentences and find their sum. The final step is to sort the sentences in inverse order of their sum. The sentences with highest frequencies summarize the text.

C. Segmentor

The Segmentor mainly works on the audio to segment the video into silent and non-silent durations. In a lecture video there exists durations which are not needed for the summarisation, like the lecturer writing on the board or general interaction with the students. All these durations have a lesser decibel value than the actual part where explanations are happening, hence we have extracted these durations by setting the minimum silent durations to be 10 seconds to avoid fragmentation and generation of too many segments. Also we are segmenting the video if the decibel value of the silent duration is lesser than or equal to -16db. Upon extracting the silent ranges we remove this from the video duration to generate non-silent ranges and use this to segment the video into individual non-silent segments.

D. Ranking Algorithm

Once the Segmentor has done its job of segmenting the video into non-silent segments the Ranking algorithm picks up these nonsilent-segments and ranks them in an order based on three important factors which are - video duration, sentence weight and word frequencies. We have assigned a score of +5 for every minute increase in the segments video duration , a +4 for every 10 percent increase in the sentence weight and a +3 for every 0.1 increase in the word frequency. Finally a sum of all these scores is generated for the non-silent video segment. Once these scores are generated, we rank the non- silent segments in order of decreasing score. We finally filter out the non-silent segments whose score is less than 50. The importance of the Ranking algorithm is to remove the segments which are essentially non-silent but at the same time do not have any useful information for the process of summarisation. An example of one such segment is when the lecturer is taking attendance.

E. Condenser

The Condenser’s job is to merge all the important seg- ments generated by the Ranking algorithm and generate a condensed video. The Condenser iterates over all the important segments and condenses them in the order of the original video, irrespective of the scores. The condensed video is a compressed version of the original video with unnecessary segments removed from it. The condensed video duration will be lesser than the original video and the order of the condensed video is the same as the original video.

F. Chalkboard Extractor

Chalkboard extractor is the most crucial module of the project. It is used to generate the handwritten lecture repre- sentations. The chalkboard representation extractor works on the condensed video to generate all the important chalkboard representations of the particular lecture video. It first generates video frames of the condensed video based on the frame rate which is decided based on the video duration of the con- densed video. Once the frames are generated, the chalkboard representations for each of these video frames are generated. The basic logic to extract chalkboard representation is to subtract the lecturer’s image from the video frame. Once all the chalkboard representations are generated for all the video frames, the final step removes all the redundant ones and the ones with zero handwritten material and mainly focuses on generating only the important ones. The important chalkboard representations are the final and the most important lecturer handwritten representations for the video lecture.

G. PDF Generator

The PDF generator gathers all the outputs from the respec- tive modules and generates a summarised handbook to be used by the students/ working professionals. The PDF generated will have all the important chalkboard representations and all the summarised text generated by the Summariser with the help of the Transcriptor. It contains one chalkboard represen- tation and a paragraph of text summary in every page so that it becomes easier for the user to correlate and understand better.
