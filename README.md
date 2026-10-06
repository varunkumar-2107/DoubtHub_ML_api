# Smart College Doubt Solving Platform

An AI/ML-powered Question & Answer community platform tailored for college students to post, discover, and resolve technical doubts efficiently.

---

## 📌 Project Overview

In a typical college technical ecosystem, students frequently encounter repetitive technical doubts. Existing communication channels often make finding past answers difficult, and multi-domain questions get mixed together.

The **Smart College Doubt Solving Platform** integrates an intelligent ML layer onto a Stack Overflow-style Q&A web application. The ML service automatically categorizes questions by domain and topic, detects duplicate or similar questions prior to posting, and recommends relevant discussion threads.

---

## 👥 Team Details — Gradient Descenders

### 👨‍🏫 Mentors
* **Harsh Giri Sir** — ML Mentor
* **Dishant Singh Sir** — ML Mentor
* **Somu Sir** — Frontend Mentor
* **Ujjawal Sir** — Backend Mentor
* **Kamakshi Rai Ma'am** — Design Mentor

### 🎓 Team Members
* **Varun Kumar** — Machine Learning
* **Divyansh Chaurasia** — Machine Learning
* **Jitendra Sahu** — Machine Learning
* **Ayush Kumar Gupta** — Machine Learning
* **Priyanka Pal** — Frontend Development
* **Abhinav Gupta** — Backend Development

---

## 🎯 Key Modules & Task Distribution

> **Suggested Work Split:** ML 40% | Frontend 30% | Backend 30%

### 🤖 Machine Learning Module (40%)
* **Automatic Classification (15%):** Predicts broad domain and topic (e.g., Frontend, Backend, ML/AI, DSA, Database, DevOps) from question titles and descriptions.
* **Similar Question Detection (15%):** Vectorizes question text to detect duplicate or highly similar questions before submission.
* **Recommendations (10%):** Recommends related questions and topics based on text similarity and user activity.

### ⚙️ Backend Module (30%)
* **Authentication & Security:** Signup, login, password hashing, and protected route access.
* **Question & Answer Pipeline:** Full CRUD operations for creating, reading, searching, and filtering doubts.
* **Voting & Reputation Engine:** Upvote/downvote logic, marking accepted answers, and updating user reputation scores.
* **ML Microservice Connection:** Handles HTTP requests and data flow between the main application and the ML service.

### 💻 Frontend Module (30%)
* **Q&A Dashboard:** Main home feed displaying latest/trending doubts with search and domain filtering.
* **Smart Similarity UI:** Real-time pop-up displaying potential duplicate questions while the user drafts a post.
* **Interactive Question Page:** Displays detailed question text, community answers, voting controls, and accepted solution indicators.
* **User Profile & Activity:** Displays student reputation, history of asked questions, given answers, and achievements.

---

## 🔄 System Architecture & Workflow

    [Student Frontend] ──> [Backend Server] ──> [ML Microservice]
            │                     │                    │
            │                     ▼                    ▼
            │             [Database Store]    [ML Models / Inference]
            │                     │
            └─────────────────────┴──> [Community Answers & Upvotes]

1. **Authentication:** Student signs up or logs into the portal.
2. **Ask a Doubt:** Student inputs title, description, and optional tags.
3. **ML Service Query:** Backend transmits draft text to the ML microservice.
4. **Classification & Similarity:** ML predicts domain/topic and computes similarity scores against existing questions.
5. **Duplicate Prevention:** If similarity crosses a threshold, existing solutions are suggested immediately.
6. **Persistence & Community:** If new, the question is stored in the database for community members to answer, upvote, and mark accepted solutions.

---

## 🛠️ Proposed Tech Stack

| Component | Stack |
| :--- | :--- |
| **Frontend** | React / Next.js / Streamlit |
| **Backend** | Node.js (Express) / Python (FastAPI/Flask) |
| **ML Service** | Python (Scikit-Learn, NLTK/SpaCy, PyTorch/TF) |
| **Database** | PostgreSQL / MongoDB / SQLite |
| **Version Control** | Git & GitHub |

---

## 🗄️ Database Schemas

    User {
      id: String
      name: String
      email: String
      role: String
      reputation: Integer
      domains: String[]
      createdAt: DateTime
    }

    Question {
      id: String
      title: String
      description: String
      authorId: String
      domain: String
      topic: String
      tags: String[]
      createdAt: DateTime
    }

    Answer {
      id: String
      questionId: String
      authorId: String
      content: String
      upvotes: Integer
      downvotes: Integer
      isAccepted: Boolean
      createdAt: DateTime
    }

---

## 🗺️ Development Roadmap

- [ ] **Phase 1:** Project setup, database architecture, and backend scaffold.
- [ ] **Phase 2:** User authentication and profile models.
- [ ] **Phase 3:** Q&A CRUD feature development.
- [ ] **Phase 4:** Voting, accepted answer logic, and search filters.
- [ ] **Phase 5:** Responsive Frontend UI & dashboard design.
- [ ] **Phase 6:** ML service infrastructure setup.
- [ ] **Phase 7:** Question domain & topic classification model training.
- [ ] **Phase 8:** Similar-question detection algorithm implementation.
- [ ] **Phase 9:** Integration between Backend and ML service endpoints.
- [ ] **Phase 10:** Recommendation engine, reputation scoring, and polish.
- [ ] **Phase 11:** Integration testing, deployment, and final presentation.

---

## 📜 License

This project is developed by team **Gradient Descenders** for the final probation task evaluation.

## Expected file structure

## Expected File Structure

```text
smart-college-doubt-platform/
│
├── frontend/                   # 💻 Frontend UI (React / Next.js)
│   ├── public/                 # Static assets (images, icons)
│   ├── src/
│   │   ├── components/         # Reusable UI parts (Navbar, QuestionCard, SimilarityPopup)
│   │   ├── pages/              # Main views (Home, AskDoubt, Profile, QuestionDetail)
│   │   ├── services/           # API calls to the Backend
│   │   ├── styles/             # CSS / Tailwind files
│   │   └── App.js              # Main React application entry
│   ├── package.json            # Frontend dependencies
│   └── .env                    # Frontend environment variables
│
├── backend/                    # ⚙️ Main Application Backend (Node.js / Express)
│   ├── src/
│   │   ├── controllers/        # Business logic (UserCtrl, QuestionCtrl, AnswerCtrl)
│   │   ├── models/             # Database schemas (User, Question, Answer)
│   │   ├── routes/             # API endpoints (/api/auth, /api/questions)
│   │   ├── middleware/         # Auth guards (JWT verification)
│   │   └── server.js           # Main backend entry point
│   ├── package.json            # Backend dependencies
│   └── .env                    # DB connection strings & JWT secrets
│
├── ml_service/                 # 🤖 Machine Learning Microservice (Python / FastAPI or Flask)
│   ├── api/
│   │   └── app.py              # ML API endpoints (/predict-domain, /check-similarity)
│   ├── models/
│   │   ├── similarity_model.pkl   # Your exported similarity model
│   │   └── classifier_model.pkl   # Your exported domain classification model
│   ├── training/
│   │   ├── train_similarity.py    # Script used to train the similarity model
│   │   └── train_classifier.py    # Script used to train the classification model
│   ├── data/
│   │   └── raw_questions.csv      # Dataset used for training (add to .gitignore)
│   └── requirements.txt        # Python dependencies (scikit-learn, pandas, flask)
│
├── .gitignore                  # Ignore node_modules, .env, and large datasets
└── README.md                   # The project overview documentation
```