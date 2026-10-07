"""
test_app.py
-----------
Automated test suite verifying all routes, APIs, and simulation triggers.
"""
import unittest
from app import app
import db


class SmartRainTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_01_login_and_auth(self):
        # 1. Unauthenticated redirect
        res = self.client.get("/")
        self.assertEqual(res.status_code, 302)

        # 2. Post login
        res = self.client.post("/login", data={"username": "demo", "password": "demo123"}, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"SmartRain", res.data)

    def test_02_pages_render(self):
        with self.client:
            self.client.post("/login", data={"username": "demo", "password": "demo123"})
            
            for endpoint in ["/", "/analytics", "/models", "/alerts", "/simulate", "/dataset"]:
                res = self.client.get(endpoint)
                self.assertEqual(res.status_code, 200, f"Failed on endpoint: {endpoint}")

    def test_03_apis(self):
        with self.client:
            self.client.post("/login", data={"username": "demo", "password": "demo123"})
            
            # Summary API
            res = self.client.get("/api/summary")
            self.assertEqual(res.status_code, 200)
            json_data = res.get_json()
            self.assertIn("status", json_data)
            self.assertIn("usage_24h", json_data)

            # Chart API
            res = self.client.get("/api/chart-data?hours=48")
            self.assertEqual(res.status_code, 200)
            chart_data = res.get_json()
            self.assertIn("labels", chart_data)
            self.assertIn("consumption", chart_data)
            self.assertIn("predicted", chart_data)
            self.assertGreater(len(chart_data["labels"]), 0)

            # Simulation API
            res = self.client.post("/api/simulate", json={"hours": 1, "leak": False})
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
