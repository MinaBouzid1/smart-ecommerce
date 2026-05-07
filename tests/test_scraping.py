import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.base_agent import BaseAgent
from agents.shopify_agent import ShopifyAgent
from agents.woocommerce_agent import WooCommerceAgent
from agents.orchestrator import run_scraping_pipeline
from utils.normalizer import normalize_products
from utils.storage import save_to_csv, save_to_sqlite


# ══════════════════════════════════════════════════════════
#  TESTS — BaseAgent
# ══════════════════════════════════════════════════════════

def test_base_agent_init():
    agent = BaseAgent(name="TestAgent", delay=2.0)
    assert agent.name == "TestAgent"
    assert agent.delay == 2.0
    assert agent.results == []


def test_base_agent_headers():
    agent = BaseAgent(name="TestAgent")
    headers = agent._get_headers()
    assert "User-Agent" in headers
    assert "Accept-Language" in headers


# ══════════════════════════════════════════════════════════
#  TESTS — ShopifyAgent
# ══════════════════════════════════════════════════════════

def test_shopify_agent_init():
    agent = ShopifyAgent(shop_urls=["https://test.com"], max_pages=3)
    assert agent.name == "ShopifyAgent"
    assert agent.shop_urls == ["https://test.com"]
    assert agent.max_pages == 3


@patch("agents.shopify_agent.requests.Session.get")
def test_shopify_scrape_empty(mock_get):
    """Test avec réponse vide."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"products": []}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    agent = ShopifyAgent(shop_urls=["https://test.com"], max_pages=1)
    result = agent.scrape()

    assert isinstance(result, list)
    assert len(result) == 0


@patch("agents.shopify_agent.requests.Session.get")
def test_shopify_scrape_with_products(mock_get):
    """Test avec produits mockés."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "products": [{
            "id": 123,
            "title": "Test Product",
            "body_html": "<p>Description</p>",
            "product_type": "Shoes",
            "vendor": "Nike",
            "tags": ["sport", "running"],
            "variants": [{
                "price": "99.99",
                "compare_at_price": "129.99",
                "available": True,
                "inventory_quantity": 50
            }],
            "created_at": "2024-01-01",
            "updated_at": "2024-01-02",
            "images": [{"src": "img.jpg"}]
        }]
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    agent = ShopifyAgent(shop_urls=["https://test.com"], max_pages=1)
    result = agent.scrape()

    assert len(result) == 1
    assert result[0]["nom"] == "Test Product"
    assert result[0]["prix"] == 99.99
    assert result[0]["remise_pct"] > 0  # compare_at_price > price
    assert result[0]["source"] == "shopify"


# ══════════════════════════════════════════════════════════
#  TESTS — WooCommerceAgent
# ══════════════════════════════════════════════════════════

def test_woocommerce_agent_init():
    agent = WooCommerceAgent(
        shop_url="https://woo.com",
        consumer_key="ck_test",
        consumer_secret="cs_test"
    )
    assert agent.name == "WooCommerceAgent"
    assert agent.auth == ("ck_test", "cs_test")


@patch("agents.woocommerce_agent.requests.Session.get")
def test_woocommerce_scrape(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{
        "id": 456,
        "name": "Woo Product",
        "description": "<p>Desc</p>",
        "short_description": "<p>Short</p>",
        "price": "49.99",
        "regular_price": "69.99",
        "sale_price": "49.99",
        "average_rating": "4.5",
        "rating_count": 120,
        "in_stock": True,
        "stock_quantity": 30,
        "total_sales": 500,
        "weight": "1.5",
        "date_created": "2024-01-01",
        "images": [{"src": "img.jpg"}],
        "categories": [{"name": "Electronics"}],
        "tags": [{"name": "promo"}]
    }]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    agent = WooCommerceAgent(
        shop_url="https://woo.com",
        consumer_key="ck",
        consumer_secret="cs"
    )
    result = agent.scrape(max_pages=1, per_page=10)

    assert len(result) == 1
    assert result[0]["nom"] == "Woo Product"
    assert result[0]["prix"] == 49.99
    assert result[0]["remise_pct"] > 0
    assert result[0]["rating"] == 4.5


# ══════════════════════════════════════════════════════════
#  TESTS — Orchestrator
# ══════════════════════════════════════════════════════════

@patch("agents.orchestrator.ShopifyAgent")
@patch("agents.orchestrator.normalize_products")
@patch("agents.orchestrator.save_to_csv")
@patch("agents.orchestrator.save_to_sqlite")
def test_run_scraping_pipeline(mock_save_sqlite, mock_save_csv, mock_normalize, mock_shopify):
    mock_agent = MagicMock()
    mock_agent.name = "ShopifyAgent"
    mock_agent.run.return_value = [{"nom": "Test", "prix": 10}]
    mock_shopify.return_value = mock_agent

    mock_normalize.return_value = pd.DataFrame([{"nom": "Test", "prix": 10}])

    config = {"shopify_urls": ["https://test.com"]}
    result = run_scraping_pipeline(config)

    assert isinstance(result, pd.DataFrame)
    mock_normalize.assert_called_once()
    mock_save_csv.assert_called_once()
    mock_save_sqlite.assert_called_once()


# ══════════════════════════════════════════════════════════
#  TESTS — Utils
# ══════════════════════════════════════════════════════════

def test_normalize_products():
    raw = [
        {"nom": "  Product A  ", "prix": "10.5", "categorie": None},
        {"nom": "Product B", "prix": 20, "categorie": "shoes"},
    ]
    df = normalize_products(raw)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["nom"].iloc[0] == "Product A"  # strip appliqué


def test_save_to_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.csv"
    save_to_csv(df, str(path))
    assert path.exists()


def test_save_to_sqlite(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.db"
    save_to_sqlite(df, str(path))
    assert path.exists()
