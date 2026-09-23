import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://elpris:elpris@localhost:5432/elpris"
)

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)

    zone = Column(String(3), nullable=False)
    target_time = Column(DateTime(timezone=True), nullable=False)
    issued_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    pred_low = Column(Float, nullable=False)
    pred_median = Column(Float, nullable=False)
    pred_high = Column(Float, nullable=False)

    actual = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("zone", "target_time", "issued_at", name="uq_prediction"),
    )


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine)
