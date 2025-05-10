import httpx
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import asyncio
from contextlib import asynccontextmanager


class APIClient:
    """
    Python client for interacting with the shopping and task management API.
    Uses httpx for HTTP requests with both synchronous and asynchronous support.
    """

    def __init__(
        self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Create sync client
        self.client = httpx.Client(
            base_url=self.base_url, headers=self.headers, timeout=self.timeout
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    @asynccontextmanager
    async def get_async_client(self):
        """
        Get an async client for making asynchronous requests.

        Usage:
            async with client.get_async_client() as async_client:
                await async_client.list_tasks_async(page=1)
        """
        async_client = httpx.AsyncClient(
            base_url=self.base_url, headers=self.headers, timeout=self.timeout
        )
        try:
            yield AsyncAPIClient(async_client, self.base_url)
        finally:
            await async_client.aclose()

    def _prepare_request_data(
        self, params: Optional[Dict] = None, data: Optional[Dict] = None
    ) -> tuple:
        """
        Prepare request parameters and data.

        Args:
            params: Query parameters
            data: Request body data

        Returns:
            Tuple of (cleaned_params, json_data)
        """
        # Remove None values from params and data
        cleaned_params = None
        if params:
            cleaned_params = {k: v for k, v in params.items() if v is not None}

        json_data = None
        if data:
            cleaned_data = {k: v for k, v in data.items() if v is not None}
            json_data = cleaned_data

        return cleaned_params, json_data

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict:
        """
        Make a synchronous HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data

        Returns:
            Response data as dictionary
        """
        cleaned_params, json_data = self._prepare_request_data(params, data)

        response = self.client.request(
            method=method, url=endpoint, params=cleaned_params, json=json_data
        )

        # Raise exception for error status codes
        response.raise_for_status()

        return response.json()

    # Task endpoints

    def list_tasks(self, page: int, is_completed: Optional[bool] = None) -> Dict:
        """
        Get a list of tasks.

        Args:
            page: Page number
            is_completed: Filter by completed flag

        Returns:
            List of tasks
        """
        params = {"page": page, "isCompleted": is_completed}

        return self._make_request("GET", "/api/tasks", params=params)

    def create_task(
        self,
        name: str,
        slug: str,
        due_date: Union[str, datetime],
        description: Optional[str] = None,
        completed: bool = False,
    ) -> Dict:
        """
        Create a new task.

        Args:
            name: Task name
            slug: Task slug
            due_date: Due date (datetime object or ISO-formatted string)
            description: Optional task description
            completed: Completion status

        Returns:
            Created task details
        """
        # Convert datetime to ISO format if needed
        if isinstance(due_date, datetime):
            due_date = due_date.isoformat()

        data = {
            "name": name,
            "slug": slug,
            "due_date": due_date,
            "description": description,
            "completed": completed,
        }

        return self._make_request("POST", "/api/tasks", data=data)

    def get_task(self, task_slug: str) -> Dict:
        """
        Get a single task by slug.

        Args:
            task_slug: Unique slug identifier for the task

        Returns:
            Task details
        """
        return self._make_request("GET", f"/api/tasks/{task_slug}")

    def delete_task(self, task_slug: str) -> Dict:
        """
        Delete a task by slug.

        Args:
            task_slug: Unique slug identifier for the task

        Returns:
            Deleted task details
        """
        return self._make_request("DELETE", f"/api/tasks/{task_slug}")

    # Shopping Item endpoints

    def get_items_by_store(
        self,
        store_slug: str,
        page: int,
        limit: Optional[int] = None,
        purchased: Optional[bool] = None,
        urgent: Optional[bool] = None,
    ) -> Dict:
        """
        Get shopping items for a specific store.

        Args:
            store_slug: Store slug
            page: Page number
            limit: Number of items per page
            purchased: Filter by purchased status
            urgent: Filter by urgent status

        Returns:
            List of shopping items for the store
        """
        params = {
            "page": page,
            "limit": limit,
            "purchased": purchased,
            "urgent": urgent,
        }

        return self._make_request(
            "GET", f"/api/items/store/{store_slug}", params=params
        )

    def search_items(self, query: str, limit: Optional[int] = 10) -> Dict:
        """
        Search for shopping items by name or text match.

        Args:
            query: Search term
            limit: Maximum number of results (1-100)

        Returns:
            Matching shopping items
        """
        params = {"query": query, "limit": limit}

        return self._make_request("GET", "/api/items/search", params=params)

    def get_urgent_items(self) -> Dict:
        """
        Get all urgent shopping items.

        Returns:
            List of urgent shopping items
        """
        return self._make_request("GET", "/api/items/list/urgent")

    def get_item(self, item_slug: str) -> Dict:
        """
        Get a single shopping item by slug.

        Args:
            item_slug: Shopping item slug

        Returns:
            Shopping item details
        """
        return self._make_request("GET", f"/api/items/{item_slug}")

    def update_item(self, item_slug: str, **kwargs) -> Dict:
        """
        Update a shopping item's attributes.

        Args:
            item_slug: Shopping item slug
            **kwargs: Any valid shopping item attributes to update

        Returns:
            Updated shopping item
        """
        valid_fields = [
            "name",
            "slug",
            "description",
            "quantity",
            "unit",
            "price",
            "currency",
            "urgent",
            "purchased",
            "purchase_date",
            "notes",
            "store_slugs",
            "category",
        ]

        # Filter out invalid fields
        data = {k: v for k, v in kwargs.items() if k in valid_fields}

        return self._make_request("PATCH", f"/api/items/{item_slug}", data=data)

    def create_item(self, name: str, slug: str, **kwargs) -> Dict:
        """
        Create a new shopping item.

        Args:
            name: Item name
            slug: Item slug
            **kwargs: Additional item attributes

        Returns:
            Created shopping item
        """
        data = {"name": name, "slug": slug, **kwargs}

        return self._make_request("POST", "/api/items", data=data)

    def mark_item_purchased(
        self,
        item_slug: str,
        purchased: bool = True,
        store_slug: Optional[str] = None,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Dict:
        """
        Mark a shopping item as purchased.

        Args:
            item_slug: Shopping item slug
            purchased: Purchase status (default True)
            store_slug: Store where item was purchased
            price: Purchase price
            quantity: Purchase quantity
            notes: Purchase notes

        Returns:
            Updated shopping item with purchase history
        """
        data = {
            "purchased": purchased,
            "store_slug": store_slug,
            "price": price,
            "quantity": quantity,
            "notes": notes,
        }

        return self._make_request(
            "PATCH", f"/api/items/{item_slug}/purchase", data=data
        )

    # Store endpoints

    def create_store(
        self,
        name: str,
        slug: Optional[str] = None,
        address: Optional[str] = None,
        notes: Optional[str] = None,
        last_visit: Optional[Union[str, datetime]] = None,
    ) -> Dict:
        """
        Create a new store.

        Args:
            name: Store name
            slug: Store slug (optional)
            address: Store address
            notes: Store notes
            last_visit: Last visit date

        Returns:
            Created store details
        """
        # Convert datetime to ISO format if needed
        if isinstance(last_visit, datetime):
            last_visit = last_visit.isoformat()

        data = {
            "name": name,
            "slug": slug,
            "address": address,
            "notes": notes,
            "last_visit": last_visit,
        }

        return self._make_request("POST", "/api/stores", data=data)

    # Chat endpoint

    def send_chat_message(
        self, message: str, conversation_id: Optional[str] = None
    ) -> Dict:
        """
        Process a chat message for natural language interaction.

        Args:
            message: User message
            conversation_id: Unique identifier for the conversation

        Returns:
            Assistant response and any actions performed
        """
        data = {"message": message, "conversation_id": conversation_id}

        return self._make_request("POST", "/api/chat", data=data)


class AsyncAPIClient:
    """
    Asynchronous API client implementation.
    """

    def __init__(self, async_client: httpx.AsyncClient, base_url: str):
        """
        Initialize async API client.

        Args:
            async_client: Async httpx client instance
            base_url: Base URL for the API
        """
        self.client = async_client
        self.base_url = base_url

    async def _make_request_async(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict:
        """
        Make an asynchronous HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data

        Returns:
            Response data as dictionary
        """
        # Remove None values from params and data
        cleaned_params = None
        if params:
            cleaned_params = {k: v for k, v in params.items() if v is not None}

        json_data = None
        if data:
            cleaned_data = {k: v for k, v in data.items() if v is not None}
            json_data = cleaned_data

        response = await self.client.request(
            method=method, url=endpoint, params=cleaned_params, json=json_data
        )

        # Raise exception for error status codes
        response.raise_for_status()

        return response.json()

    # Async Task endpoints

    async def list_tasks_async(
        self, page: int, is_completed: Optional[bool] = None
    ) -> Dict:
        """
        Get a list of tasks asynchronously.

        Args:
            page: Page number
            is_completed: Filter by completed flag

        Returns:
            List of tasks
        """
        params = {"page": page, "isCompleted": is_completed}

        return await self._make_request_async("GET", "/api/tasks", params=params)

    async def create_task_async(
        self,
        name: str,
        slug: str,
        due_date: Union[str, datetime],
        description: Optional[str] = None,
        completed: bool = False,
    ) -> Dict:
        """
        Create a new task asynchronously.

        Args:
            name: Task name
            slug: Task slug
            due_date: Due date (datetime object or ISO-formatted string)
            description: Optional task description
            completed: Completion status

        Returns:
            Created task details
        """
        # Convert datetime to ISO format if needed
        if isinstance(due_date, datetime):
            due_date = due_date.isoformat()

        data = {
            "name": name,
            "slug": slug,
            "due_date": due_date,
            "description": description,
            "completed": completed,
        }

        return await self._make_request_async("POST", "/api/tasks", data=data)

    async def get_task_async(self, task_slug: str) -> Dict:
        """
        Get a single task by slug asynchronously.

        Args:
            task_slug: Unique slug identifier for the task

        Returns:
            Task details
        """
        return await self._make_request_async("GET", f"/api/tasks/{task_slug}")

    async def delete_task_async(self, task_slug: str) -> Dict:
        """
        Delete a task by slug asynchronously.

        Args:
            task_slug: Unique slug identifier for the task

        Returns:
            Deleted task details
        """
        return await self._make_request_async("DELETE", f"/api/tasks/{task_slug}")

    # Async Shopping Item endpoints

    async def get_items_by_store_async(
        self,
        store_slug: str,
        page: int,
        limit: Optional[int] = None,
        purchased: Optional[bool] = None,
        urgent: Optional[bool] = None,
    ) -> Dict:
        """
        Get shopping items for a specific store asynchronously.

        Args:
            store_slug: Store slug
            page: Page number
            limit: Number of items per page
            purchased: Filter by purchased status
            urgent: Filter by urgent status

        Returns:
            List of shopping items for the store
        """
        params = {
            "page": page,
            "limit": limit,
            "purchased": purchased,
            "urgent": urgent,
        }

        return await self._make_request_async(
            "GET", f"/api/items/store/{store_slug}", params=params
        )

    async def search_items_async(self, query: str, limit: Optional[int] = 10) -> Dict:
        """
        Search for shopping items by name or text match asynchronously.

        Args:
            query: Search term
            limit: Maximum number of results (1-100)

        Returns:
            Matching shopping items
        """
        params = {"query": query, "limit": limit}

        return await self._make_request_async("GET", "/api/items/search", params=params)

    async def get_urgent_items_async(self) -> Dict:
        """
        Get all urgent shopping items asynchronously.

        Returns:
            List of urgent shopping items
        """
        return await self._make_request_async("GET", "/api/items/list/urgent")

    async def get_item_async(self, item_slug: str) -> Dict:
        """
        Get a single shopping item by slug asynchronously.

        Args:
            item_slug: Shopping item slug

        Returns:
            Shopping item details
        """
        return await self._make_request_async("GET", f"/api/items/{item_slug}")

    async def update_item_async(self, item_slug: str, **kwargs) -> Dict:
        """
        Update a shopping item's attributes asynchronously.

        Args:
            item_slug: Shopping item slug
            **kwargs: Any valid shopping item attributes to update

        Returns:
            Updated shopping item
        """
        valid_fields = [
            "name",
            "slug",
            "description",
            "quantity",
            "unit",
            "price",
            "currency",
            "urgent",
            "purchased",
            "purchase_date",
            "notes",
            "store_slugs",
            "category",
        ]

        # Filter out invalid fields
        data = {k: v for k, v in kwargs.items() if k in valid_fields}

        return await self._make_request_async(
            "PATCH", f"/api/items/{item_slug}", data=data
        )

    async def create_item_async(self, name: str, slug: str, **kwargs) -> Dict:
        """
        Create a new shopping item asynchronously.

        Args:
            name: Item name
            slug: Item slug
            **kwargs: Additional item attributes

        Returns:
            Created shopping item
        """
        data = {"name": name, "slug": slug, **kwargs}

        return await self._make_request_async("POST", "/api/items", data=data)

    async def mark_item_purchased_async(
        self,
        item_slug: str,
        purchased: bool = True,
        store_slug: Optional[str] = None,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Dict:
        """
        Mark a shopping item as purchased asynchronously.

        Args:
            item_slug: Shopping item slug
            purchased: Purchase status (default True)
            store_slug: Store where item was purchased
            price: Purchase price
            quantity: Purchase quantity
            notes: Purchase notes

        Returns:
            Updated shopping item with purchase history
        """
        data = {
            "purchased": purchased,
            "store_slug": store_slug,
            "price": price,
            "quantity": quantity,
            "notes": notes,
        }

        return await self._make_request_async(
            "PATCH", f"/api/items/{item_slug}/purchase", data=data
        )

    # Async Store endpoints

    async def create_store_async(
        self,
        name: str,
        slug: Optional[str] = None,
        address: Optional[str] = None,
        notes: Optional[str] = None,
        last_visit: Optional[Union[str, datetime]] = None,
    ) -> Dict:
        """
        Create a new store asynchronously.

        Args:
            name: Store name
            slug: Store slug (optional)
            address: Store address
            notes: Store notes
            last_visit: Last visit date

        Returns:
            Created store details
        """
        # Convert datetime to ISO format if needed
        if isinstance(last_visit, datetime):
            last_visit = last_visit.isoformat()

        data = {
            "name": name,
            "slug": slug,
            "address": address,
            "notes": notes,
            "last_visit": last_visit,
        }

        return await self._make_request_async("POST", "/api/stores", data=data)

    # Async Chat endpoint

    async def send_chat_message_async(
        self, message: str, conversation_id: Optional[str] = None
    ) -> Dict:
        """
        Process a chat message for natural language interaction asynchronously.

        Args:
            message: User message
            conversation_id: Unique identifier for the conversation

        Returns:
            Assistant response and any actions performed
        """
        data = {"message": message, "conversation_id": conversation_id}

        return await self._make_request_async("POST", "/api/chat", data=data)


# Example usage
if __name__ == "__main__":
    # Synchronous example
    with APIClient(base_url="https://api.example.com") as client:
        # List tasks
        tasks = client.list_tasks(page=1, is_completed=False)
        print(f"Found {len(tasks['series']['result']['tasks'])} tasks")

        # Create a task
        new_task = client.create_task(
            name="Buy groceries",
            slug="buy-groceries",
            due_date=datetime.now(),
            description="Get items for dinner",
        )
        print(f"Created task: {new_task['series']['result']['task']['name']}")

    # Asynchronous example
    async def async_example():
        client = APIClient(base_url="https://api.example.com")

        async with client.get_async_client() as async_client:
            # List tasks asynchronously
            tasks = await async_client.list_tasks_async(page=1)
            print(
                f"Found {len(tasks['series']['result']['tasks'])} tasks asynchronously"
            )

            # Search for items asynchronously
            search_results = await async_client.search_items_async(
                query="Milk", limit=5
            )
            print(
                f"Found {len(search_results['result']['items'])} items matching 'Milk' asynchronously"
            )

            # Run multiple requests concurrently
            tasks_result, urgent_items_result = await asyncio.gather(
                async_client.list_tasks_async(page=1),
                async_client.get_urgent_items_async(),
            )

            print(
                f"Retrieved {len(tasks_result['series']['result']['tasks'])} tasks and "
                f"{len(urgent_items_result['success']['result']['items'])} urgent items concurrently"
            )

    # Run the async example
    if __name__ == "__main__":
        asyncio.run(async_example())
