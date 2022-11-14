
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="">
    <img src="images\alicerun_compressed.gif" alt="Logo" width="540" 
height="540">
  </a>

  <h2 align="center"> ALICERUN  </h2>

  <h4 align="center">
    Analize your digital activity's, mood, habits and predict what <br />
to do next is best for your work-life balance 
<br /><br />
OR <br />

How to mix TODO lists, tracking work sessions,<br />
habit and mood tracking, computer and smartphone tracking,<br /> CBT 
techniques, analyze them and build <br />
recommendation system for task prioritization and balanced life 
    <br /><br />
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

[//]: # (    <a href="">View Demo</a>)

[//]: # (    ·)

[//]: # (    <a href="">Report Bug</a>)

[//]: # (    ·)

[//]: # (    <a href="">Request Feature</a>)
  </h4>
</div>





<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)



[//]: # (<p align="right">&#40;<a href="#top">back to top</a>&#41;</p>)


##  :beginner: Motivation

This year is total mess, a lot of frustration.
So i decided to make a tool for quantitative self-assessment to track the emotional background and the progress of 
life goals, habits, and mood. Engineer, huh.
  
* So I wanted to find (or return) the source of motivation, to concentrate on my emotional condition. 
* If it turns out to be useful for someone else (open source) - cool. 
* Also to draw attention to myself as a programmer, to find a good job.

Contact me on [![LinkedIn][linkedin-shield]][linkedin-url]



## The idea came from the logic of how digital products work.
![](https://i.vas3k.club/3183b26bc1611740309c5993ac0ecb181bf11bd27d7450d263281571e6ab2b40.jpg)

The product must have metric(s) and a tracking process to adjust performance.

What is a great work-life balance? People feel it as a normal state of health, a balanced emotional state, etc. We do not measure this state - the metrics are not digitized. It turns out that today I feel good, but I don’t remember how it was a week ago. What about a monthly period? Is there any progress? It's not clear.
 
**Questions running through my head**
  
	* How can I objectively measure the quality of my work?
	* How to understand that work-life balance is good?
	* How to understand what is missing for life drive?
	* In what areas do I fade, and on which should I focus my efforts in the first place?
	* What patterns of behavior prevail?
	* Which of them should have been worked out yesterday? (addition to therapy)

**There was interest, and excitement in front of an incomprehensible task.**

 ![](images\1.png)
 * The task was to develop a system of metrics, analytics for tracking them, and prediction.
 * As a result, I wanted to analyze progress every day and predict (with the help of machine learning) the most 
   suitable tasks (for work and life), and habits for the best balance between life goals.


## :wrench: Briefly about the sources of inspiration. 
Here are some self-digitalization pet projects that inspired my R&D process.
  
* [Post by Stephen Wolfram](https://writings.stephenwolfram.com/2012/03/the-personal-analytics-of-my-life/). 10 years this article has been an example for me, um, of "cyberpunk awareness" and possession of analytics tools.
* Here is more  [up to date experiment from Felix Krause](https://howisfelix.today/)

 ![Formula of Motivation](https://i.vas3k.club/3863eeacbacda0bf50b911bf2bc7ada51c3473ef3f566d28b52cdea086dbdab4.jpg)
* [ Alex Vermeer inspired me in terms of breaking down motivation into components](https://alexvermeer.com/getmotivated/). When I first saw the [motivation formula](https://en.wikipedia.org/wiki/Temporal_motivation_theory) I felt relieved.
* And finally, [Simon Chabot](https://framagit.org/chabotsi/miband2_analysis) analysed miband data, his repo was 
  huge finding. 

##  :wrench: Screenshots of dashboard and ready to use features 
![Final cut, №1](https://raw.githubusercontent.com/mldsqc/alicerun/main/images/2.png)
![Final cut, №2](https://raw.githubusercontent.com/mldsqc/alicerun/main/images/3.png)
![Final cut, №3](https://raw.githubusercontent.com/mldsqc/alicerun/main/images/4.png)
  
![ecosystem in blocks](https://i.vas3k.club/6a8d038ecb3ea15f7b793d7e7f7f142768553e4493275429ba30bfc2f34efaba.jpg)
And step by step:
	
        * Data collection, synchronization, and archiving
        * Personal storage server
        * Data analysis
        * Predicting the best task schedule
        * Visualization in the calendar
        * Dashboard with metrics and graphs

 ![Metrics system](https://i.vas3k.club/43db3b3232f463adbe2a2744e372db63fc797dd4705613b2cd3d39728767d867.jpg)

 Metrics system:

* register habits and emotions several times a day.
* we attribute them to each of the vital areas: FINANCIAL, CAREER, EMOTIONAL, PHYSICAL, FUN_RECREATION, INTELLECTUAL, 
  PARTNER, SOCIAL
* enter the value of the assessment\contribution for each action. And counting the total value of personal growth. This is optional. As a result, I settled on a radial chart (upper-left from the very first picture) as the main graph to look at first.


###  :zap: What data/where is it collected from?
![Integration](images\5.png)
  
We collect the data from these sources:
	* Emotions and habits via Telegram bot, 4 times a day, reminders via Google calendar pushes
	* Actions on the computer from ActivityWatch
	* Android Activity from ActivityWatch, ActionDash. With root - we immediately take SQL files
	* Date from smartwatch(Miband4) from GadgetBridge
	* To-do lists and tasks from the Microsoft ToDo application (via the API - it's easier to immediately SQL-files)
	* Work sessions are tracked in Toggl (using the API)


###  :electric_plug: Installation
What will you  need:

- Hosting of your choise(i've used Google cloud)
- Telegram keys, Microsoft Graph API keys, Toggl API key
- Registration of personal Google cloud app (to get access to drive, calendar, etc.)


## :zap: What is a technological stack 
  
stack: pandas, PostgreSQL, time-series forecasting, KNN clusterisation,
plotly, dash, streamlit, toggl API, Microsoft graph API, google cloud, telegram-bot
  
## :zap: Disclaimer, after which the project could not even start.

* There are problems with registering emotions and habits in real-time. We all have instruments to register a pulse, 
stress level, and EEG in everyday life. I wish somebody will come up with small and easily accessible sensors for reading emotions instantly (if there are recommendations, write *), yeap, I know about reading emotions by face with CV, and i think it's not ideal yet).
* Therefore, in the project it works like this: in the period from 11 to 14, I experienced this list of emotions, and 
  made these habits. And not: 11-01 - emotion No. 11, 11-49 - habit No. 2 .... That means, we are registering cumulatively. This point then limits the forecasting methods.
 
![Oh no](https://i.vas3k.club/ba16791faf11c4ee75fa3d90a547ac922162049b688b587ad12e47d8a53fbf68.jpg)

 And here's the **most important**:
 * The fact that I experienced this emotion and/or performed this habit does not guarantee that it will affect the progress of a particular task. That is, there is no evidence of a causal relationship (and how can this be verified in principle?).
 * Therefore, I am a simple Buddhist monk - watching the storm and spinning data in pandas.

###  :package: What hard skills i learned

* It was very interesting to face the problems of data format organization. Questions from the series: what storage structure to take to build such a graph after processing-analysis
* The toughest thing I remember was 2 days spent on 15 lines of code processing data in a specific type of storage. Frustration over the edge - but - this is how steel was tempered)))

* I started tracking data to analytics, but it still turned out to be a small amount of data to learn on. Partly because sometimes I did not break tasks into small subtasks. For example, I did 1 big task for 3 weeks and did not register subtasks inside it. Lost data. So I had to run on simple ranking at the beginning for the cold-start problem, lol.
* Due to the small amount of data collected (on completed tasks), the choice of prediction algorithms was limited. I took KNN clustering(with accuracy as the metric) to predict the execution time of tasks (I don’t use other metrics yet). Played with ARIMA for time series of emotions, without any hope.


 ![Description of metrics from the task dataset](https://i.vas3k.club/76e555c1b1a047be3f952046cf4e974347ab8efa478bbbd98e080f0cd47322cd.jpg)

### :package: How much did i spend?
![project track](https://i.vas3k.club/006c4e46ea73f6728177351144c893876b0c0f443b6d8c09d2d23ad9c88c36fb.jpg)

Spent 240 hours. Lines of code ~ 8200, of which 3500 (experiments) did not enter production, from the rest ~ 15% of someone else's code (smartwatch block)




## :zap: Open track of future features
  
	* Introduce mechanics from CBT. Questionnaires - to encourage writing 5-minute journals. You can then pick up the answers using NLP (natural language processing)
	* Analysis of spending budget. automatic micro-investment (or charity) in crypto for unfulfilled goals as a punishment
	* Analysis of behavior on the phone and desktop is not screwed yet, but the data is integrated. And think about what features can be built on this data.
	* Integrate Anki cards


## :star2: Don't forget to write about the name
alicerun is a reference to Lewis Carroll.
*My dear, here we must run as fast as we can, just to stay in place. And if you wish to go anywhere you must run twice as fast as that.

other options were: Munchausen (a reference to pulling himself by the hair from the mud), moospace (like mootivation), well, quite a punkish JORK... maybe mr. Jork Munhausen can be fine? I dunno

## :star2: How does the system helps me?
I will not say that I open the dashboard every day. But I use the entire tracking workflow every day and this 
creates the effect of self-discipline. I began to notice behavioral patterns more often.
  
The best thing that has been achieved is a cozy feeling of control over the situation. 

[//]: # ()
[//]: # (###  :hammer: Build)

[//]: # (Write the build Instruction here.)

[//]: # ()
[//]: # (### :rocket: Deployment)

[//]: # (Write the deployment instruction here.)

[//]: # ()
[//]: # ( ###  :fire: Contribution)

[//]: # ()
[//]: # ( Your contributions are always welcome and appreciated. Following are the things you can do to contribute to this project.)

[//]: # ()
[//]: # ( 1. **Report a bug** <br>)

[//]: # ( If you think you have encountered a bug, and I should know about it, feel free to report it [here]&#40;&#41; and I will take care of it.)

[//]: # ()
[//]: # ( 2. **Request a feature** <br>)

[//]: # ( You can also request for a feature [here]&#40;&#41;, and if it will viable, it will be picked for development.  )

[//]: # ()
[//]: # ( 3. **Create a pull request** <br>)

[//]: # ( It can't get better then this, your pull request will be appreciated by the community. You can get started by picking up any open issues from [here]&#40;&#41; and make a pull request.)

[//]: # ()
[//]: # ( > If you are new to open-source, make sure to check read more about it [here]&#40;https://www.digitalocean.com/community/tutorial_series/an-introduction-to-open-source&#41; and learn more about creating a pull request [here]&#40;https://www.digitalocean.com/community/tutorials/how-to-create-a-pull-request-on-github&#41;.)

[//]: # ()
[//]: # ( ### :cactus: Branches)

[//]: # ()
[//]: # ( I use an agile continuous integration methodology, so the version is frequently updated and development is really fast.)

[//]: # ()
[//]: # (1. **`stage`** is the development branch.)

[//]: # ()
[//]: # (2. **`master`** is the production branch.)

[//]: # ()
[//]: # (3. No other permanent branches should be created in the main repository, you can create feature branches but they should get merged with the master.)

[//]: # ()
[//]: # (**Steps to work with feature branch**)

[//]: # ()
[//]: # (1. To start working on a new feature, create a new branch prefixed with `feat` and followed by feature name. &#40;ie. `feat-FEATURE-NAME`&#41;)

[//]: # (2. Once you are done with your changes, you can raise PR.)

[//]: # ()
[//]: # (**Steps to create a pull request**)

[//]: # ()
[//]: # (1. Make a PR to `stage` branch.)

[//]: # (2. Comply with the best practices and guidelines e.g. where the PR concerns visual elements it should have an image showing the effect.)

[//]: # (3. It must pass all continuous integration checks and get positive reviews.)

[//]: # ()
[//]: # (After this, changes will be merged.)


[//]: # (## :star2: Credit/Acknowledgment)

[//]: # ()
[//]: # (* [Choose an Open Source License]&#40;https://choosealicense.com&#41;)

[//]: # (* [GitHub Emoji Cheat Sheet]&#40;https://www.webpagefx.com/tools/emoji-cheat-sheet&#41;)

[//]: # (* [Malven's Flexbox Cheatsheet]&#40;https://flexbox.malven.co/&#41;)

[//]: # (* [Malven's Grid Cheatsheet]&#40;https://grid.malven.co/&#41;)

[//]: # (* [Img Shields]&#40;https://shields.io&#41;)

[//]: # (* [GitHub Pages]&#40;https://pages.github.com&#41;)

[//]: # (* [Font Awesome]&#40;https://fontawesome.com&#41;)

[//]: # (* [React Icons]&#40;https://react-icons.github.io/react-icons/search&#41;)





<p align="right">(<a href="#top">back to top</a>)</p>

[//]: # (##  :lock: License)

[//]: # (Add a license here, or a link to it.)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat&logo=appveyor
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat&logo=appveyor
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat&logo=appveyor
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat&logo=appveyor
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat&logo=appveyor
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat&logo=appveyor&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/
[product-screenshot]: images/2.png



[//]: # ()
[//]: # ()
[//]: # (<!-- TABLE OF CONTENTS -->)

[//]: # ( <h3><div align="center">)

[//]: # (<details>)

[//]: # ( <summary><u><b>Table of Contents</b></u></summary> )

[//]: # (  <ol>)

[//]: # (    <li>)

[//]: # (      <a href="#about-the-project">About The Project</a>)

[//]: # (      <ul>)

[//]: # (        <li><a href="#built-with">Built With</a></li>)

[//]: # (      </ul>)

[//]: # (    </li>)

[//]: # (    <li>)

[//]: # (      <a href="#getting-started">Getting Started</a>)

[//]: # (      <ul>)

[//]: # (        <li><a href="#prerequisites">Prerequisites</a></li>)

[//]: # (        <li><a href="#installation">Installation</a></li>)

[//]: # (      </ul>)

[//]: # (    </li>)

[//]: # (    <li><a href="#usage">Usage</a></li>)

[//]: # (    <li><a href="#roadmap">Roadmap</a></li>)

[//]: # (    <li><a href="#contributing">Contributing</a></li>)

[//]: # (    <li><a href="#license">License</a></li>)

[//]: # (    <li><a href="#contact">Contact</a></li>)

[//]: # (    <li><a href="#acknowledgments">Acknowledgments</a></li>)

[//]: # (  </ol>)

[//]: # (</details> </div></h3>)



[//]: # ()
[//]: # (### :notebook: results of analytics)

[//]: # ()
[//]: # (- most interesting  - analysis of behavioral patterns)

[//]: # (- or self psychotherapy by yourself)

[//]: # ()
[//]: # (- emotional risk management with ml prediction and )

[//]: # ()




[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (## :ledger: Index)

[//]: # ()
[//]: # (- [Motivation]&#40;#beginner-motivation&#41;)

[//]: # (- [Data sources]&#40;#zap-data-sources&#41;)

[//]: # (  - [Installation]&#40;#electric_plug-installation&#41;)

[//]: # (  - [Commands]&#40;#package-commands&#41;)

[//]: # (- [Development]&#40;#wrench-development&#41;)

[//]: # (  - [Pre-Requisites]&#40;#notebook-pre-requisites&#41;)

[//]: # (  - [Development Environment]&#40;#nut_and_bolt-development-environment&#41;)

[//]: # (  - [File Structure]&#40;#file_folder-file-structure&#41;)

[//]: # (  - [Build]&#40;#hammer-build&#41;  )

[//]: # (  - [Deployment]&#40;#rocket-deployment&#41;  )

[//]: # (- [Community]&#40;#cherry_blossom-community&#41;)

[//]: # (  - [Contribution]&#40;#fire-contribution&#41;)

[//]: # (  - [Branches]&#40;#cactus-branches&#41;)

[//]: # (  - [Guideline]&#40;#exclamation-guideline&#41;  )

[//]: # (- [FAQ]&#40;#question-faq&#41;)

[//]: # (- [Resources]&#40;#page_facing_up-resources&#41;)

[//]: # (- [Gallery]&#40;#camera-gallery&#41;)

[//]: # (- [Credit/Acknowledgment]&#40;#star2-creditacknowledgment&#41;)

[//]: # (- [License]&#40;#lock-license&#41;)
