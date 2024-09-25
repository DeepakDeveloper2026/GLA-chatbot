import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables (Google API key)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Embed your static data into the chatbot directly
gla_data = """
1.About this project is GLA Chatbot Using LLM
Project Overview
The "GLA Chatbot Using LLM" project aims to develop a sophisticated chatbot for the Global Learning Academy 
(GLA) utilizing Large Language Models (LLMs). The chatbot will provide students, faculty, and staff with an 
intelligent assistant capable of answering questions, providing learning resources, and facilitating administrative 
tasks.Leveraging the capabilities of LLMs, the chatbot will offer natural language understanding and generation, 
enhancing the overall user experience.
Objectives
1. Student Support: Provide instant assistance to students on course-related queries, resource access, and administrative issues.
2. Faculty Assistance: Aid faculty members in managing course content, answering questions about academic policies, and facilitating communication with students.
3. Administrative Efficiency: Streamline administrative processes by automating routine tasks and inquiries.
4. Enhanced Learning: Offer personalized learning recommendations and resources based on user queries and interactions.
Key Components
1. Large Language Model (LLM): Use an advanced LLM such as GPT-4 to power the chatbot’s natural language understanding and generation capabilities.
2. User Interface (UI): Develop a user-friendly interface accessible via web and mobile platforms to interact with the chatbot.
3. Backend Integration: Integrate with GLA’s existing databases and systems to provide accurate and realtime information.
4. Context Management: Implement context management to handle multi-turn conversations and provide relevant responses based on the ongoing dialogue.
5. Analytics and Reporting: Incorporate analytics to track usage patterns, common queries, and chatbot performance, enabling continuous improvement.
Technical Approach
1. LLM Integration
o Utilize GPT-4 or a similar LLM to power the chatbot.
o Train the LLM on GLA-specific data to improve its accuracy and relevance in providing responses.
2. User Interface Development
o Design a conversational UI that is intuitive and easy to use for both students and faculty.
o Ensure the UI is responsive and works seamlessly across various devices.
3. Backend Integration:
o Connect the chatbot to GLA’s learning management system (LMS), student information system (SIS), and other relevant databases.
o Implement secure APIs to facilitate data exchange between the chatbot and GLA systems.
4. Context Management
o Develop mechanisms to maintain conversation context and handle follow-up questions.
o Use session management to track user interactions and provide contextually relevant responses.
5. Natural Language Processing (NLP)
o Implement NLP techniques to interpret and process user queries effectively.
o Use intent recognition to understand the purpose of the user's questions and provide accurate answers.
6. Analytics and Reporting
o Set up analytics tools to monitor chatbot interactions and gather insights.
o Use these insights to refine the chatbot’s performance and improve response accuracy.
Benefits
 Instant Support: Provides immediate assistance to users, enhancing the overall experience.
 Efficiency: Reduces the workload on administrative and academic staff by automating routine tasks and queries.
 Personalized Learning: Offers tailored learning resources and recommendations, supporting individual learning needs.
 24/7 Availability: Ensures users can access support and information at any time, without time constraints.
Challenges
 Data Privacy: Ensuring the security and privacy of user data, especially when handling sensitive 
information.
 Accuracy: Maintaining high accuracy in responses, particularly for complex or nuanced queries.
 Integration Complexity: Integrating the chatbot with existing systems and databases without disrupting ongoing processes.
 Continuous Improvement: Regularly updating the LLM with new data and refining its algorithms to keep up with evolving user needs.
Conclusion
The "GLA Chatbot Using LLM" project aims to revolutionize the way students, faculty, and staff interact with 
the Global Learning Academy’s systems and resources. By leveraging the capabilities of Large Language Models, 
the chatbot will provide intelligent, context-aware assistance, making academic and administrative processes more 
efficient and user-friendly. This initiative will not only enhance user experience but also contribute to a more 
responsive and adaptive learning environment at GLA.
GLA University Information
Address: 17km Stone, NH-19, Mathura-Delhi Road, P.O. Chaumuhan, Mathura-281 406 (U.P.), India.
Contact Numbers
 +91-5662-250900
 +91-5662-250909
 +91-5662-241489
 +91-9927064017
Admission Helpline: 9027068068
Email: glauniversity@gla.ac.in
About GLA University
"I like the dreams of the future better than the history of the past.” - Thomas Jefferson
In 1991, Shri Narayan Das Agrawal decided to fulfill his father's dream, the Late Shri Ganeshi Lal Agrawal, of 
establishing an institute for quality education for the people and the region & beyond. This led to the foundation 
of a milestone at the karmabhoomi of Lord Krishna. GLA University has been actively involved with social causes 
since its inception and has drawn appreciation for its work in various facets of societal paradigms.
Vision: We envision ourselves as a pace-setting university of Academic Excellence focused on education, 
research, and development in established and emerging professions.
Mission
 To impart quality professional education, conduct commendable research, and provide credible 
consultancy and extension services per current and emerging socio-economic needs.
 To continuously enhance and enrich the teaching/learning process and set standards that other institutes 
would want to emulate.
 To be student-centric, promoting the overall growth and development of intellect and personality of our 
students, making our alums worthy citizens and highly sought-after professionals worldwide.
 To empower faculty and staff members, fostering an ambiance of harmony, mutual respect, cooperative 
endeavor, and receptivity towards positive ideas.
 To proactively seek regular feedback from all stakeholders and take appropriate measures based on them, 
leading to an excellent learning process.
Rankings & Accolades
 Ranked #2 in UP & #6 in INDIA amongst all Private B-Schools Ranking 2018 by Best Private University 
in UP in Engineering by Survey 2018.
 Rated ‘AAA’ amongst India’s Best Engineering Colleges 2020.
 Ranked #1 in INDIA amongst the Top Emerging Engineering Institutes in survey 2018 by Times of 
India.
 Ranked #1 University in UP by Best Private University in UP and also in North, East & North Eastern 
India.
 Ranked #3 in UP amongst Top 75 B Schools - BBA Edition survey 2018.
 National Employability Award among the Top 10% engineering campuses nationally 2019.
 Ranked #3 in UP Institute of Business Management - BBA.
 Best Private University in UP Survey 2019 & 2020 by Dainik Jagran.
 Awarded Excellence in Placements & Alumni Network among best private universities in India.
 University Grant Commission (UGC) 12B Status by UGC Among the top private universities in India to 
receive this honor.
 ARIIA Band Excellent Award for blending innovation in learning and beyond classroom experiential 
activities.
 NAAC ‘A+’ Grade for high standard Infrastructure, Learning Resources, Research, Innovation, etc.
 ACTIVE Local Chapter by NPTEL Ranked among the top 100 institutions with Local Chapter tag by 
NPTEL.
 IACBE Accreditation Specialized accreditation through the International Accreditation Council for 
Business Education.
 NIRF RANKED Institute of Pharmaceutical Research is ranked as the 54th best institute for pharmacy 
education in India by NIRF.
 Times Higher Education Rankings 2024: All India Rank 44, Worldwide Rank Band 1001-1200, Research 
Quality (World) Rank 691.
Events
 NEP-2020 and Teaching and Learning (21-Jun-24)
 Uttar Pradesh State Karate (08-Jun-24)
 SPANDAN 2024 (05-Apr-24)
 Parv Awadh ki Holi (Holi Milan) (16-Mar-24)
 Aagaaz
Graduate Courses
 B.Tech (Computer Science & Engineering)
 B.Tech (Hons.) B.Tech CSE Honours (AI and Analytics) in Partnership with Intel and NEC
 B.Tech Electronics & Communication Engineering
 B.Tech EC (With Minor in Computer Science)
 B.Tech EC (With Specialization in VLSI)
 B.Tech Electrical & Electronics Engineering
 B.Tech (Electrical Engineering)
 B.Tech Electrical Engineering With Minor in CS
 B.Tech Electrical Engineering (With Specialization in Electric Vehicle Technology)
 B.Tech (Mechanical Engineering)
 B.Tech Mechanical Engineering (With Minor in CS)
 B.Tech Mechanical Engineering (Specialization in Automobile)
 B.Tech Mechanical Engineering (Specialization in Mechatronics)
 B.Tech (Civil Engineering)
 B.Tech (Biotechnology)
 B.Tech Computer Science & Engineering in Partnership with Microsoft (specialization with AIML)
 B.Tech Mechanical Engineering (Specialization in Smart Manufacturing)
 BCA / BCA (Hons./ By Research)
 B.Tech Electronics & Communications (Lateral Entry)
 B.Tech Electronics & Computer Engineering
 B.Tech Mechanical Engineering (Lateral Entry)
 B.Com Global Accounting with CIMA
 BBA (Management Science)
 BBA/ BBA (Hons./ By Research)
 B.Tech CSE (Lateral Entry)
 B.Tech Civil Engineering (Lateral Entry)
 B.Tech Electrical Engineering (Lateral Entry)
 B. Pharm
 B. Pharm (Lateral Entry)
 BBA (Family Business)
 B.Com/B.Com (Hons./By Research)
 B.A. Economics / B.A. Economics (Hons./ By Research)
 B.A. English / B.A. English (Hons./ By Research)
 BCA with Specialization in Data Science
 B.Sc. Biotech/ B.Sc. Biotech (Hons./ By Research)
 B.Sc (Hons.) Agriculture
 B.Sc. Chemistry/ B.Sc. Chemistry (Hons./ By Research)
 B.Sc. Physics/ B.Sc. Physics (Hons./ By Research)
 B.Sc. Mathematics/ B.Sc. Mathematics (Hons./ By Research) with Specialization in Data Science
 B.A. LLB (Hons.)
 BBA LLB (Hons.)
 B. Com L.L.B (Hons.)
 B.Ed.
 Bachelor of Library and Information Science
Admission Procedure
 Visit the GLA University website and click on "Admission Procedure".
 For GLAET Registration, Online Test Fee Payment, and Online GLAET Slot Booking, click on the 
respective links.
 Candidates can book test slots, view results, and deposit the token amount for academic and hostel fees 
online.
 For further details on the admission process and fees, visit the admission portal or contact the helpline 
number.
For any further queries, you can call 9027068068.
1. Undergraduate B.Tech. Programs:- B.Tech. in Computer Science (CS)- 4 years- Fees: 1,85,000 per 
year- B.Tech. in Electronics & Communication (EC)- 4 years- Fees: 1,76,000 per yearSpecializations/Minors: VLSI, Computer Engineering, Electrical Engineering,
Mechanical Engineering- B.Tech. in CS (Specialization: Artificial Intelligence and Machine Learning)- 4 
years
Fees: 2,20,000 per year- B.Tech. in Mechanical Engineering (ME)- 4 years- Fees: 2,15,000 per yearB.Tech. (Hons.) in Computer Science & Engineering (Specialization: AI and Analytics)- 4
years- Fees: 2,35,000 per year- Lateral Entry options available for various branches- Duration: 3 
years- No fees
2. Postgraduate M.Tech. Program:- M.Tech. with various specializations- 2 years- Fees: 1,76,000 per 
year
3. Undergraduate B.Pharm. Program:- B.Pharm.- 4 years- Fees: 1,51,000 per year- Lateral Entry 
option available- Duration: 3 years- No fees
4. Other Undergraduate Programs:- BBA Management Science- 3 years- Fees: 1,35,000 per year- BBA 
(Family Business)- 3 years- Fees: 1,35,000 per year- B.Com. GA With CIMA- 3 years- Fees: 1,35,000 
per year- BBA/BBA (Hons./By Research)- 3/4 years- Fees: 1,05,000 per year- B.Com./B.Com. 
(Hons./By Research)- 3/4 years- Fees: 1,05,000 per year- B.A. Economics/B.A. Economics (Hons./By 
Research)- 3/4 years- Fees: 71,000 per
year- B.A. English/B.A. English (Hons./By Research)- 3/4 years- Fees: 44,000 per year- BCA With 
Specialization in Data Science- 3 years- Fees: 1,35,000 per year- BCA/BCA (Hons./By Research)- 3/4 
years- Fees: 1,05,000 per year- B.Sc. Bio-Tech/B.Sc. Biotech (Hons./By Research)- 3/4 years- Fees: 
1,05,000 per year- B.Sc. Math/B.Sc. Math (Hons./By Research) with Specialization in Data Science3/4
years- Fees: 44,000 per year- B.Sc. (Hons.) Agriculture- 4 years- Fees: 79,000 per year- B.Sc. 
Chemistry/B.Sc. Chemistry (Hons./By Research)- 3/4 years- Fees: 44,000 per
year- B.Sc. Physics/B.Sc. Physics (Hons./By Research)- 3/4 years- Fees: 44,000 per year- B.A. LLB 
(Hons.)- 5 years- Fees: 1,35,000 per year- B.Com. LLB (Hons.)- 5 years- Fees: 1,35,000 per year- BBA. 
LLB (Hons.)- 5 years- Fees: 1,35,000 per year- B.Ed.- 2 years- Fees: 53,000 per year- Diploma in Library 
and Information Sciences- 1 year- Fees: 40,000 per year- Bachelor of Library and Information 
Sciences- 1 year- Fees: 60,000 per year
Postgraduate Programs
- M.A. English- 2 years- Fees: 48,000 per year- MBA-Business Analytics- 2 years- Fees: 2,65,000 per 
year- MBA (Hons.)- 2 years- Fees: 2,65,000 per year- MBA- 2years- Fees: 2,20,000 per year- MBA 
(LSCM)- 2 years- Fees: 2,60,000 per year- MBA (FMB)- 2 years- Fees: 2,60,000 per year- MBA in 
Construction Management- 2 years- Fees: 1,76,000 per year- MCA- 2years- Fees: 1,43,000 per yearM.Sc. (Biotech)- 2 years- Fees: 1,26,000 per year- M.Sc. (Microbiology & Immunology)- 2 years- Fees: 
1,26,000 per year- M.Sc. (Bioinformatics)- 2 years- Fees: 1,26,000 per year- M.Sc. (Chemistry)- 2 
years- Fees: 53,000 per year- M.Sc. (Physics)- 2 years- Fees: 53,000 per year- M.Sc. (Mathematics)- 2 
years- Fees: 53,000 per year- M.Sc. (Agriculture)- 2 years- Fees: 82,000 per year- LLM- 1 year- Fees: 
1,40,000 per year- Master of Library and Information Science- 1 year- Fees: 70,000 per year
Diploma Programs
Diploma Engineering- 3 years- Fees: 65,000 per year- Diploma (Lateral Entry)- 2 years- No fees
7. Doctoral Programs:- Ph.D. (Other Specializations)- 3 years- Fees: 70,000 per year- Ph.D. (Law, 
Management, Pharmacy)- 3 years- Fees: 90,000 per year
"Gla received visits from a diverse range of companies in last few years, including Berger
Paints, British Telecom, DAMCO, EduKyu, Google, Indus Valley Partners, Fiserv, Farelabs,
Finstem Group, Guardian, Faber, Infoedge, Jair Network & Services, Khimji Ramdas,
Luminous, Mphasis, Koyo, MRF, KSolves, Oracle, Orange Business Services, Panasonic,
Reynolds, Shopclues, Sify, Sparx, SpringWorks, Tolexo, uCertify, Ujjivan, Vinove, Virtusa,
Webkul, Cyber Group, Daffod, Deloitte, DigiValet, Directi, Everest, Gemini Solution, Global
Logic Innovation by Design, Goldman Sachs, Hashedin, Indiamart, Jaro Education, JK
Paper Ltd, Juspay, Justdial, Pinnacle, Sagitec, Sopra Steria, Playdude, Squareboat,
Technovert, Tek System, UnitedHealth Group, Voltas, Lifecell, Innodata, Tech Mahindra,
Cryoviva, Sun Pharma, Leapcraft, Ingenious e-Brain, Lawyered, Anthem, Allied Market
Research, IntelliPaat, and Akums."GLA University don’t mention tracks in report card but yes
at the end of their degree we can provide them with a certification which states this as Micro
specialization domain certification.
Top Placements: Here are the top highlights of placements at GLA University:- Vidhan Sharma, a 
B.Tech graduate in Computer Science, secured a position at Njuma
Consulting in 2023 with a salary of 55 lakh per annum.- Radhika Singh, also holding a B.Tech degree 
in Computer Science, was hired by VMware
in 2022 with an annual package of 19.97 lakh.- Deepesh Pandey, another B.Tech graduate in 
Computer Science, found employment at
Above Awe with a yearly salary of 28.04 lakh.- Adarsh Bhardwaj, an MBA graduate, secured a 
position at V2U with a package of 19.8 lakh
per annum.- Ashwani Saraswat, holding an MBA degree, joined V2U in 2023 with an annual package 
of
19.8 lakh.- Kushagra Ganwar, a B.Tech graduate in Computer Science, secured a position at Amazon
in 2022 with an annual salary of 44 lakh.- Pranjul Maheswari, a B.Tech graduate in Computer Science, 
was placed at Paloalto in
2024 with a package of 41 lakh per annum.- Gopal Mishra, a B.Tech graduate specialised in 
Electronics and Communication
Engineering, landed a position at HLS Asia Limited in 2024 with a salary of 12.5 lakh per
annum.- Adarsh Kumar, an MBA graduate, secured a position at Posterity in 2024 with an annual
package of 8 lakh.- Prashant Duve, an MBA graduate, secured a position at Bajaj in 2024 with a 
salary of 6
lakh per annum.
Placement Showcase: These placements showcase the excellence and success of GLA University's 
graduates in
the professional world.
1. Report Cards: Tracks are not mentioned in report cards, but students can receive a Micro
Specialization Domain Certification at the end of their degree.
2. Tracks: A track is a collection of subjects. For example, if a student chooses the Gaming
track, they will study subjects related to gaming from the 3rd to 8th semester, including a
project.
3. Track Selection: Once a track is selected in the 3rd semester, it remains chosen for the
entire duration from the 2nd to 4th year.
4. Scholarship Criteria: The scholarship criteria for B.Tech CS (H) is based on PCM marks
only.
5. Study Abroad: Students have the opportunity to study at IIT Gandhinagar for one
semester based on IIT-G's criteria.
6. Value-Added Courses: GLA offers a wide range of value-added courses in both in-class
and online study modes.
7. Industry Tie-Ups: GLA has significant tie-ups with companies such as Intel, NEC, IBM,
WIPRO, Tech-Mahindra, and Global Logic, among others.
8. Flexibility in Programs: The university offers the provision of a Minor program, allowing
students to opt for minors in Management, Robotics and Automation, Electronics
Engineering, etc.
9. Alumni Network: GLA has a large alumni network of over 35,000 individuals, leveraging
them for various purposes including Board of Studies, industry talks, workshops, and
internship and placement opportunities.
10. Collaboration with Foreign Universities: Students have the opportunity to study at
universities abroad where GLA has collaborations.
11. Support for Innovation and Entrepreneurship: GLA encourages and supports innovation
and entrepreneurship through various programs, courses, incubators, accelerators,
hackathons, collaborative spaces, networking events, funding, and grants.
12. Center of Excellence (CoE): CoEs at GLA aid students in acquiring varied technical skills
through specialized training, hands-on learning, industry collaboration, expert guidance,
research opportunities, networking events, and continuous learning resources.
13. Differentiating Programs: The B.Tech AIML program is described as premium, while the
B.Tech (Hons) program in partnership with Intel NEC is considered the flagship program,
with differences in dual specialization, industry-based training, better exposure,
personalized mentoring, and expert lectures.
14. Placement Assistance: GLA assists students before placement drives through mock
tests, mock interviews, group discussions, comprehensive viva for lab subjects, and
guest lectures by industry experts.
15. Specialization Courses: Students can choose from the following tracks/specializations for
Core CSE starting from the 2nd year
● AI&DataScience
● AIDriven DevOps
● AI&CyberSecurity
● ARVR&Gaming
● CloudComputing
● Full Stack Development
● IoT&Robotics
● Vanilla CSE (No Specific Track)
● Industry Oriented Learning and Specializations
Students have the flexibility to select a track based on their interests and career goals, allowing
them to specialize in a particular area within the field of Computer Science and Engineering.
1. B.Tech CSE (Core):
● Credits: 160-165
● Eligibility: Minimum 50% in PCM
● FeeStructure: 1.85L, 2.0 L, 2.15L, 2.3L (year wise)
2. B.Tech CSE with Specialization in AIML:
● Credits: 180-185
● Eligibility: Minimum 60% in Physics, Mathematics, and Engineering/Computer Science
● FeeStructure: 2.20L, 2.25L, 2.30L, 2.35L (year wise)
3. B.Tech CSE (Honors) in AI & Analytics with INTEL & NEC:
● Credits: 180-185
● Eligibility: Minimum 75% in Physics, Mathematics, and Engineering/Computer Science
● FeeStructure: 2.35L, 2.40L, 2.45L, 2.50L (year wise)
Each program has different credit requirements, eligibility criteria, and fee structures, catering to
students with varying academic backgrounds and interests within the field of Computer Science
and Engineering.
GLA University is a private university in Mathura, Uttar Pradesh. It has been declared fit to
receive central assistance under Section 12B of UGC Act, 1956 after proper assessment for the
same by the UGC. It is recognized by University Grants Commission (UGC), NCTE and
Pharmacy Council of India. It has been accredited by the National Assessment and Accreditation
Council (NAAC) with ‘A+’ Grade.
Institutes:
1. Institute of Engineering and Technology
2. Institute of Applied Science and Humanities
3. Institute of Business Management
4. Institute of Pharmaceutical Research
5. University of Polytechnic
6. Faculty of Education
7. Institute of Legal Studies & Research
8. Physical Education
9. Institute of Agriculture
● GLAUniversity, Mathura Motto
ऋत ााना न मि त
● Motto in English
Without knowledge there is no salvation
Type
Private Established 2010; 14 years ago Affiliation UGC Chancellor Narayan Das Agrawal Location 
Chaumuhan,
Mathura, Uttar Pradesh, India Campus Rural Website gla.ac.in
● Established
2010; 14 years ago
● Affiliation
UGC
● Chancellor
Narayan Das Agrawal A businessman who founded GLA University in Mathura, Uttar Pradesh, India 
in 1991 to fulfill his father's dream of establishing a quality institute for education. Agrawal is the 
president of the university's society.
● CEO Of GLA
Neeraj Agrawal
● Location
 Chaumuhan, Mathura, Uttar Pradesh, India
 Campus:Rural
Website: gla.ac.in
About GLA University
 Rankings and Surveys
o Ranked #2 in UP amongst all Private B-Schools in 2018 by Best Private University in 
UP.
o Best Private University in UP in Engineering by Survey 2018.
o Rated ‘AAA’ amongst India’s Best Engineering Colleges 2020.
o Ranked #1 in India amongst the Top Emerging Engineering Institutes in 2018 by 
Times of India.
o Ranked #1 University in UP by Best Private University in UP and also in North, East & 
North Eastern India.
o Ranked #3 in UP amongst Top 75 B-Schools (BBA Edition) survey 2018.
o National Employability Award among the Top 10% engineering campuses nationally 
in 2019.
o Ranked #3 in UP Institute of Business Management (BBA) by Best Private University 
in UP Survey 2019 & 2020 by Dainik Jagran.
o Awarded Excellence in Placements & Alumni Network among best private 
universities in India.
 Accreditations and Awards
o University Grant Commission (UGC) 12B Status.
o ARIIA Band Excellent Award for blending innovation in learning and beyond 
classroom experiential activities.
o NAAC ‘A+’ Grade for high standard Infrastructure, Learning Resources, Research, 
Innovation, etc.
o Active Local Chapter by NPTEL, ranked among the top 100 institutions with Local 
Chapter tag.
o Specialized accreditation through the International Accreditation Council for 
Business Education (IACBE).
o NIRF Ranked: Institute of Pharmaceutical Research is ranked as the 54th best 
institute for pharmacy education in India by NIRF.
Times Higher Education Rankings 2024
o All India Rank: 44
o Worldwide Rank Band: 1001-1200
o Research Quality (World) Rank: 691
Achieving Career Goals in Data Analytics Develop a Strong Foundation
● Acquire deep understanding of statistics, programming languages like Python or R, and
data manipulation techniques.
● Invest time in learning about databases, data visualization, and machine learning
algorithms.
Pursue Continuous Learning
● Embraceamindset of continuous learning through industry conferences, workshops, and
webinars.
● Explore online courses and certifications to sharpen skills and gain recognized
credentials.
Gain Practical Experience
● Seekopportunities for hands-on experience through internships, projects, or freelance
work.
● Participate in data analytics competitions or contribute to open-source projects.
Build a Strong Network
● Connect with professionals, attend industry events, and join online communities.
● Engagein discussions, seek advice, and share insights for career growth opportunities.
Cultivate Soft Skills
● Develop communication, critical thinking, problem-solving, and teamwork skills.
● Effective communication helps convey complex ideas to non-technical stakeholders.
Specialize in a Niche
● Consider specializing in a specific subfield like business intelligence or data visualization.
● Deepenexpertise in relevant tools and techniques to differentiate yourself in the job
market.
Seek Mentorship
● Lookfor experienced professionals who can provide valuable guidance and insights.
● Actively seek mentorship opportunities through professional networks or online platforms.
Stay Updated with Industry Trends
● Follow industry thought leaders, subscribe to relevant blogs, and join data analytics
communities.
● Adaptyour skill set according to emerging technologies and industry best practices.
By following these strategies and focusing on continuous growth and development, you can
position yourself for success in the dynamic field of data analytics.
Faculty at GLA University
1. Prof. Durg Singh Chauhan - Pro Chancellor
2. Mr. Ankit Prakash - Assistant Professor
3. Prof. Phalguni Gupta - Vice-Chancellor and Professor
4. Mr. Suresh Pratap Singh - Director- T&D
5. Prof. Ashok Bhansali - Professor and Dean
6. Prof. Ashish Sharma - Professor, Dean Academics, and Chief Proctor
7. Prof. Dilip Kumar Sharma - Professor & Dean of International Relations & Academic 
Collaborations
8. Prof. Anand Singh Jalal - Professor
9. Mr. Nishant Singh - Assistant Professor
10. Dr. Hitendra Garg - Professor & Associate Head
11. Dr. Mayank Srivastava - Associate Professor
12. Dr. Neeraj Gupta - Associate Head & Associate Professor
13. Prof. Kumar Sankar Ray - Distinguished Professor
14. Prof. Samir Kumar Bandhopadhyay - Distinguished Professor
15. Mr. Sachin Sharma - Assistant Professor
16. Dr. Rakesh Kr. Galav - Associate Professor
17. Prof. Amitava Sen - Distinguished Professor
18. Mr. Anjani Kumar Rai - Assistant Professor
19. Dr. Ashish Sharma - Associate Professor
20. Ms. Anupam Yadav - Assistant Professor
21. Mr. Narendra Mohan - Assistant Professor
22. Dr. Subhash Chand Agrawal - Associate Professor
23. Dr. Rahul Pradhan - Associate Professor
24. Dr. Sandeep Rathor - Associate Head & Associate Professor
25. Dr. Rajesh Kumar Tripathi - Associate Professor
26. Mr. Mayank Agrawal - Assistant Professor
27. Dr. Saurabh Singhal - Associate Professor
28. Dr. Anshy Singh - Assistant Professor & Associate Director- Software Development Cell
29. Dr. Praveen Mittal - Associate Professor
30. Mr. Piyush Vashistha - Assistant Professor
31. Dr. Aditya Saxena - Assistant Professor
32. Dr. Vinod Jain - Assistant Professor
33. Dr. Pooja Pathak - Associate Professor
34. Mr. Akash Yadav - Assistant Professor
35. Dr. Himanshu Sharma - Associate Professor & Dean- Student Welfare
36. Mr. Asheesh Tiwari - Assistant Professor
37. Mr. Mohd. Amir Khan - Assistant Professor
38. Mr. Dhirendra Prasad Yadav - Assistant Professor
39. Dr. Ajitesh Kumar - Associate Professor & Deputy Chief Proctor
40. Ms. Mona Kumari - Assistant Professor
41. Mr. Mandeep Singh - Assistant Professor
42. Dr. Madhu Sudan Kumar - Assistant Professor
43. Dr. Aradhya Kumar Shukla - Assistant Professor
44. Dr. Neetu Faujdar - Position not specified
45. Mr. Navin Kumar Agrawal - Assistant Professor
46. Mr. Ankur Sisodia - Assistant Professor
47. Dr. Ashish Srivastava - Assistant Professor
48. Mr. Raushan Kumar Singh - Assistant Professor
49. Mr. Umakant Ahirwar - Assistant Professor
50. Dr. Abhishek Sharma - Assistant Professor
51. Ms. Paromita Goswami - Assistant Professor
52. Dr. Manish Gupta - Associate Professor
53. Dr. Swati Srivastava - Assistant Professor
54. Ms. Sweta Singh - Assistant Professor
55. Dr. Premnarayan Arya - Assistant Professor
56. Mr. Shelesh Krishna Saraswat - Assistant Professor
57. Mr. Dheeraj Kalra - Assistant Professor
58. Mrs. Divya Singh - Assistant Professor
59. Mr. P. Bachan - Assistant Professor
60. Mr. Deepak Mittal - Assistant Professor
61. Mrs. Neetu Agrawal - Assistant Professor
62. Mrs. Alka Agrawal - Assistant Professor
63. Mr. Rohit Agarwal - Assistant Professor
64. Mr. Girijapati Sharma - Assistant Professor
65. Mr. Sachendra Singh Chauhan - Assistant Professor
66. Dr. Manu Banga - Assistant Professor
67. Ms. Ankita Chauhan - Assistant Professor
68. Ms. Priya Chaudhary - Assistant Professor
69. Mr. Bhupendra Kumar Saraswat - Assistant Professor
70. Dr. Law Kumar Singh - Assistant Professor
71. Mr. Varun Mishra - Assistant Professor
72. Mr. Prashant Singh - Assistant Professor
73. Ms. Deepti Agrawal - Assistant Professor
74. Mr. Mridul Dixit - Assistant Professor
75. Ms. Priya Shrivastava - Assistant Professor
76. Ms. Apurva Garg - Assistant Professor
77. Mr. Ronak Agrawal - Assistant Professor
78. Ms. Saumya Tripathi - Assistant Professor
79. Mr. Vineesh Kumar Singh - Assistant Professor
80. Ms. Arti Badhoutiya - Assistant Professor
81. Mr. Abhinav Khatri - Assistant Professor
82. Ms. Kirti Agrawal - Assistant Professor
83. Mr. Arvind Prasad - Assistant Professor
84. Ms. Sonali Agrawal - Assistant Professor
85. Mrs. Neetu Agrawal - Assistant Professor
86. Mrs. Alka Agrawal - Assistant Professor
87. Mr. Rohit Agarwal - Assistant Professor
88. Mr. Girijapati Sharma - Assistant Professor
89. Mr. Sachendra Singh Chauhan - Assistant Professor
90. Dr. Manu Banga - Assistant Professor
91. Ms. Ankita Chauhan - Assistant Professor
92. Ms. Priya Chaudhary - Assistant Professor
93. Mr. Bhupendra Kumar Saraswat - Assistant Professor
94. Dr. Law Kumar Singh - Assistant Professor
95. Mr. Varun Mishra - Assistant Professor
96. Mr. Prashant Singh - Assistant Professor
Hostels in GLA
Thousands of students from all across the country live in the hostels in GLA University, which comes 
closest to being 'a home away from home'. GLA University provides well-furnished hostel facility to its 
outstation candidates. The campus hostels are surrounded by lush green lawns and playing fields. The 
separate facilities for boys and girls, caring wardens and a tight security ensures a pleasant stay allowing 
you to focus on your academics. The hostels have complete power backup. Facilities for indoor games like 
TT, carom, chess, Television etc. are also available.
A warden / hostel in-charge looks after the administration of the hostel. Students enjoy a homely and 
comfortable stay with a sense of camaraderie and fraternity. Hygienic, quality food prepared by 
professionally-qualified cooks is provided to the students in the hostel canteen. Meals provided in the 
hostel mess are wholesome and nourishing. The mess caters to the tastes of the students with varied 
culinary preferences from different regions. The mess menu is planned and managed by the students in 
consultation with the caterer and the management.
Students also get a highly professional on-campus laundry service. Whilst internet access and STD 
facilities ensure close contact with family and friends outside, the fully equipped common rooms allow for 
that relaxed time with friends on Campus.
The campus has strict security through smart cards, biometric readers, latest IP cameras, fire warning 
systems, 24-hour guards. For total care for in any emergencies, we have our own dispensaries with best 
doctors and nurses.
We have 15 boys’ hostels with a housing capacity of 4300+ boys and 4 girls’ hostel with a seating capacity 
of 1400 girls.
Students are provided TV, Newspapers etc. all the hostels have in-door games like, Carom, Badminton, 
TT. A multi-purpose Gym has been set-up for complete fitness which include cardio & strength training.
A newly-constructed 800-room hostel at GLA has been awarded the 'best 11 architecture designs' by the 
World Architecture Community, UK – a globally-acknowledged architecture consortium. This hostel 
consists of five, 4-level blocks. Its structure depicts the streets of a historical place. The hostel building is 
oriented towards the north to give residents a scenic garden view. .
Each room has a wedge-shaped bay window that provides protection from the sun and lets in natural light 
at the same time. Ventilation openings onto the internal corridors facilitate cross-ventilation that does away 
with the need for a climate control system in a zone where the average temperature is over 30° C for 8 
months of the year.
"""

def get_text_chunks(text):
    # Split the static text into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Embed the text chunks into a vector store using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Define the prompt for answering questions based on the context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    # Use Google's ChatGoogleGenerativeAI for the LLM model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # Reload the vector store and find relevant documents based on user input
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Use the conversational chain to generate a response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Output the response formatted in bullet points
    reply = response["output_text"]
    
    # Format the response into bullet points
    bullet_points = reply.splitlines()
    formatted_reply = "\n".join(f"- {point.strip()}" for point in bullet_points if point.strip())

    # Output the response using Markdown for bullet points
    # st.markdown("**Reply:**")
    st.markdown(formatted_reply)

def main():
    # Configure Streamlit page
    st.set_page_config("Chat PDF")
    st.header("GLA ChatBot Help & Support 💁")

    # User input section
    user_question = st.text_input("Ask a Question from the GLA Data")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Process Data"):
            with st.spinner("Processing..."):
                # Instead of processing PDF, process the static GLA data
                text_chunks = get_text_chunks(gla_data)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
