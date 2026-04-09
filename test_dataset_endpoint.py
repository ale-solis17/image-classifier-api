import unittest
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api.routes import router
from app.db.models import Image
from app.db.session import get_session
from fastapi.staticfiles import StaticFiles


class DatasetEndpointTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(self.engine)

        with Session(self.engine) as session:
            now = datetime.now(UTC)
            session.add(
                Image(
                    file_path="data/uploads/one.jpg",
                    original_name="one.jpg",
                    predicted_label="Bacteroides fragilis",
                    confidence=0.91,
                    status="labeled",
                    human_label="Bacteroides fragilis",
                    created_at=now,
                )
            )
            session.add(
                Image(
                    file_path="data/uploads/two.jpg",
                    original_name="two.jpg",
                    predicted_label="Staphylococcus aureus",
                    confidence=0.88,
                    status="pending",
                    human_label=None,
                    created_at=now - timedelta(minutes=1),
                )
            )
            session.commit()

        app = FastAPI()
        app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")
        app.include_router(router)

        def override_get_session():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_session] = override_get_session
        self.client = TestClient(app)

    def test_list_dataset_returns_items_and_total(self):
        response = self.client.get("/dataset")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["limit"], 100)
        self.assertEqual(payload["offset"], 0)
        self.assertEqual(len(payload["items"]), 2)
        self.assertEqual(payload["items"][0]["original_name"], "one.jpg")
        self.assertTrue(payload["items"][0]["image_url"].endswith("/uploads/one.jpg"))

    def test_list_dataset_filters_by_status(self):
        response = self.client.get("/dataset?status=labeled")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["status"], "labeled")

    def test_list_dataset_supports_pagination(self):
        response = self.client.get("/dataset?limit=1&offset=1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["limit"], 1)
        self.assertEqual(payload["offset"], 1)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["original_name"], "two.jpg")
        self.assertTrue(payload["items"][0]["image_url"].endswith("/uploads/two.jpg"))


if __name__ == "__main__":
    unittest.main()
