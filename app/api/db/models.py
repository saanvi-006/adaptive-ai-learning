from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Performance(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)

    correct = Column(Integer)
    wrong = Column(Integer)
    total = Column(Integer)
    accuracy = Column(Float)

    last_intent = Column(String)
    intent_stats = Column(JSON)
    weak_intents = Column(JSON)


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True)
    user_id = Column(String)

    total_answered = Column(Integer)
    correct = Column(Integer)
    wrong = Column(Integer)
    accuracy = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)