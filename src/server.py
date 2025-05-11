from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import asyncio
from typing import List, Dict, Literal, Optional, Any

# Import our async API client
from api import APIClient, AsyncAPIClient


@dataclass
class ApiContext:
    """
    Context for API client access.

    This class provides a structured way to access API clients within the MCP framework.
    It's made available to tools through the MCP context object.

    Attributes:
        sync_client: The synchronous API client instance for making API requests
        async_client: Optional asynchronous API client instance for making async API requests

    Usage in tools:
        The ApiContext is accessible through ctx.request_context.lifespan_context in tool functions:

        @mcp.tool()
        async def my_tool(ctx: Context):
            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            # Use the sync client directly, or get an async client
            async with client.get_async_client() as async_client:
                # Make async API calls
                response = await async_client.some_method_async()
    """

    sync_client: APIClient
    async_client: Optional[AsyncAPIClient] = None


@asynccontextmanager
async def api_lifespan(server: FastMCP) -> AsyncIterator[ApiContext]:
    """
    Manage API client lifecycle with type-safe context.

    This lifecycle handler initializes and properly disposes of the API client,
    ensuring connections are properly managed.

    The MCP Context (ctx) object in tool functions provides important utilities:

    1. Logging and progress reporting:
       - ctx.info(message): Log an informational message
       - ctx.warn(message): Log a warning message
       - ctx.error(message): Log an error message
       - ctx.report_progress(current, total): Report progress during long operations

    2. Access to API context:
       - ctx.request_context.lifespan_context: Access the API context with clients

    3. Request metadata:
       - ctx.request_context.request_id: Unique ID for the current request
       - ctx.request_context.user_id: ID of the user making the request (if available)
       - ctx.request_context.timestamp: When the request was made

    4. Response modification:
       - ctx.add_metadata(key, value): Add custom metadata to the response

    Example usage in a tool:
        @mcp.tool()
        async def my_tool(param: str, ctx: Context):
            # Log information about the operation
            await ctx.info(f"Processing request with param: {param}")

            # Report progress during long operations
            await ctx.report_progress(0, 100)

            # Access the API client
            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            # Perform operations
            # ...

            # Add custom metadata to the response
            ctx.add_metadata("operation_time_ms", 150)

            # Return results
            return json.dumps({"result": "success"})
    """
    # Initialize clients on startup
    api_url = server.server_config.get("api_url", "http://localhost:8787")
    api_key = server.server_config.get("api_key")

    # Create synchronous client
    sync_client = APIClient(base_url=api_url, api_key=api_key)

    # Track async client if we'll need it
    async_client = None

    try:
        # Yield the context to the server
        yield ApiContext(sync_client=sync_client)
    finally:
        # Cleanup on shutdown
        # The sync client will be closed by its __exit__ method
        pass


class AsyncMCPTasksShoppingServer:
    """
    MCP server for Tasks & Shopping API with asynchronous operation support.

    This server implements a Machine Comprehension Protocol (MCP) interface for the
    Tasks & Shopping API, providing a set of tools that language models can use to
    interact with the API in a structured way. The server supports asynchronous
    operations for better performance with concurrent requests.

    Features:
    - Comprehensive set of tools for shopping management (items, stores)
    - Task management capabilities (create, list, update tasks)
    - Asynchronous API client support for concurrent operations
    - Connection lifespan management for reliable API access
    - Proper error handling and structured responses
    - Progress reporting for long-running operations

    The server can operate in different transport modes:
    - stdio: Standard input/output for direct CLI interaction or testing
    - sse: Server-Sent Events for web-based interfaces
    - streamable-http: HTTP streaming for more complex integrations

    Tools provided:
    - get_store_items: Retrieve items associated with a specific store
    - get_urgent_items: Get all items marked as urgent
    - get_item: Get detailed information about a specific item
    - search_items: Search for items by text query
    - create_item: Create a new shopping item
    - create_store: Create a new store
    - mark_item_purchased: Mark an item as purchased
    - batch_operations: Execute multiple operations in parallel
    """

    def __init__(self, api_url: str, api_key: Optional[str] = None, port=8000):
        """
        Initialize the MCP server for the Tasks & Shopping API with async support.

        Args:
            api_url: Base URL for the API (e.g., "http://localhost:8787" or "https://api.example.com")
            api_key: Optional API key for authentication
            port: Port number to use for server when using SSE transport (default: 8000)

        Example:
            server = AsyncMCPTasksShoppingServer(
                api_url="https://api.example.com",
                api_key="your-api-key",
                port=8080
            )
            server.run("stdio")  # Run using standard I/O transport
        """
        # Create the MCP server with lifespan support
        self.mcp = FastMCP(
            "Tasks-Shopping-API",
            lifespan=api_lifespan,
            port=port,
        )
        self.mcp.server_config = {}

        # Register resources
        self.register_resources()

        # Register tools
        self.register_tools()

    def register_resources(self):
        """
        Register all resource-related tools for the MCP server.

        This method registers tools primarily focused on retrieving and querying
        existing data from the API. Resource tools are generally read-only and
        used for retrieving information rather than modifying it.

        Registered tools:
        - get_store_items: Retrieve shopping items for a specific store
        - get_urgent_items: Get all urgent shopping items across all stores
        - get_item: Get a single shopping item by its slug
        - search_items: Search for shopping items using a text query
        """

        @self.mcp.tool()
        async def get_store_items(
            store_slug: str,
            page: int,
            ctx: Context,
            limit: Optional[int] = None,
            purchased: Optional[bool] = None,
            urgent: Optional[bool] = None,
        ) -> str:
            """
            Get shopping items for a specific store.

            This tool retrieves a paginated list of shopping items associated with a specific store.
            Results can be filtered by purchase status and urgency.

            Args:
                store_slug: The unique identifier for the store (e.g., "walmart", "target")
                page: The page number for pagination (starts at 1)
                ctx: The MCP context object for logging and accessing the API client
                limit: Optional maximum number of items to return per page
                purchased: Optional filter for purchased status (True/False)
                urgent: Optional filter for urgent items (True/False)

            Returns:
                A JSON string containing the paginated list of shopping items for the store

            Example response structure:
                {
                  "success": {
                    "result": {
                      "items": [
                        {
                          "slug": "milk",
                          "name": "Milk",
                          "description": "Whole milk",
                          "quantity": 1,
                          "unit": "gallon",
                          "price": 3.99,
                          "currency": "USD",
                          "urgent": false,
                          "purchased": false,
                          "store_slugs": ["grocery-store"]
                        },
                        ...
                      ],
                      "pagination": {
                        "page": 1,
                        "total_pages": 5,
                        "total_items": 47
                      }
                    }
                  }
                }
            """
            await ctx.info(f"Fetching items for store: {store_slug} (page: {page})")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                response = await async_client.get_items_by_store_async(
                    store_slug=store_slug,
                    page=page,
                    limit=limit,
                    purchased=purchased,
                    urgent=urgent,
                )
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def get_urgent_items(ctx: Context) -> str:
            """
            Get all urgent shopping items.

            This tool retrieves a list of all shopping items marked as urgent across all stores.
            It's a convenient way to quickly find high-priority items that need attention.

            Args:
                ctx: The MCP context object for logging and accessing the API client

            Returns:
                A JSON string containing the list of all urgent shopping items

            Example response structure:
                {
                  "success": {
                    "result": {
                      "items": [
                        {
                          "slug": "eggs",
                          "name": "Eggs",
                          "description": "Organic free-range eggs",
                          "quantity": 1,
                          "unit": "dozen",
                          "price": 5.99,
                          "currency": "USD",
                          "urgent": true,
                          "purchased": false,
                          "store_slugs": ["grocery-store", "farmers-market"]
                        },
                        ...
                      ],
                      "count": 3
                    }
                  }
                }
            """
            await ctx.info("Fetching urgent items")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                response = await async_client.get_urgent_items_async()
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def get_item(item_slug: str, ctx: Context) -> str:
            """
            Get a single shopping item by slug.

            This tool retrieves detailed information about a specific shopping item
            using its unique slug identifier.

            Args:
                item_slug: The unique identifier for the shopping item (e.g., "milk", "eggs")
                ctx: The MCP context object for logging and accessing the API client

            Returns:
                A JSON string containing the detailed information for the specified item

            Example response structure:
                {
                  "success": {
                    "result": {
                      "item": {
                        "slug": "eggs",
                        "name": "Eggs",
                        "description": "Organic free-range eggs",
                        "quantity": 1,
                        "unit": "dozen",
                        "price": 5.99,
                        "currency": "USD",
                        "urgent": true,
                        "purchased": false,
                        "purchase_date": null,
                        "purchase_history": [],
                        "notes": null,
                        "store_slugs": ["grocery-store", "farmers-market"],
                        "category": "dairy",
                        "created_at": "2023-01-15T12:30:45Z",
                        "updated_at": "2023-01-16T09:15:30Z"
                      }
                    }
                  }
                }

            Error response when item not found:
                {
                  "error": {
                    "message": "Item not found",
                    "code": "NOT_FOUND",
                    "status": 404
                  }
                }
            """
            await ctx.info(f"Fetching item: {item_slug}")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                response = await async_client.get_item_async(item_slug)
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def search_items(
            ctx: Context, query: str, limit: Optional[int] = 10
        ) -> str:
            """
            Search for shopping items by name or text match.

            This tool allows searching for shopping items using a text query.
            It performs a case-insensitive search against item names, descriptions,
            and other relevant fields. Results are limited to a maximum number of items.

            Args:
                ctx: The MCP context object for logging and accessing the API client
                query: The search term to look for in items (e.g., "milk", "fruit")
                limit: Maximum number of results to return (default: 10, range: 1-100)

            Returns:
                A JSON string containing the search results matching the query

            Example response structure:
                {
                  "success": {
                    "result": {
                      "items": [
                        {
                          "slug": "almond-milk",
                          "name": "Almond Milk",
                          "description": "Unsweetened almond milk",
                          "quantity": 1,
                          "unit": "quart",
                          "price": 3.99,
                          "currency": "USD",
                          "urgent": false,
                          "purchased": false,
                          "store_slugs": ["grocery-store"]
                        },
                        {
                          "slug": "whole-milk",
                          "name": "Whole Milk",
                          "description": "Organic whole milk",
                          "quantity": 1,
                          "unit": "gallon",
                          "price": 4.99,
                          "currency": "USD",
                          "urgent": false,
                          "purchased": false,
                          "store_slugs": ["grocery-store"]
                        },
                        ...
                      ],
                      "count": 2,
                      "query": "milk"
                    }
                  }
                }

            Notes:
                - The search is performed across multiple fields including name, description, and notes
                - Results may include partial matches (e.g., searching for "milk" might return "almond milk")
                - The limit parameter restricts the number of results returned
            """
            await ctx.info(f"Searching for items: {query} (limit: {limit})")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                response = await async_client.search_items_async(
                    query=query, limit=limit
                )
                return json.dumps(response, indent=2)

    def register_tools(self):
        """
        Register all action-related tools for the MCP server.

        This method registers tools that perform actions and modify data in the API.
        These tools include creating new items, updating existing items, and
        performing batch operations.

        Registered tools:
        - create_item: Create a new shopping item with specified attributes
        - create_store: Create a new store where items can be purchased
        - mark_item_purchased: Mark an item as purchased with purchase details
        - batch_operations: Execute multiple operations in parallel for better performance

        Action tools are distinguished from resource tools (registered in register_resources)
        by their ability to create, modify, or delete data within the API, rather than
        just querying existing information.
        """

        @self.mcp.tool()
        async def create_item(
            name: str,
            slug: str,
            ctx: Context,
            description: Optional[str] = None,
            quantity: Optional[float] = None,
            unit: Optional[str] = None,
            price: Optional[float] = None,
            currency: str = "MXN",
            urgent: bool = False,
            store_slugs: Optional[List[str]] = None,
            category: Optional[str] = None,
        ) -> str:
            """
            Create a new shopping item.

            This tool creates a new shopping item with the specified attributes.
            The item will be added to the shopping database and can be associated
            with one or more stores.

            Args:
                name: Item name (e.g., "Milk", "Bread")
                slug: Item slug (unique identifier, e.g., "milk", "bread") - must be URL-friendly
                ctx: The MCP context object for logging and accessing the API client
                description: Optional detailed description of the item
                quantity: Optional numeric quantity needed (e.g., 2, 0.5)
                unit: Optional unit of measurement (e.g., "kg", "pieces", "liters")
                price: Optional expected or estimated price
                currency: Currency code for the price (default: "MXN")
                urgent: Whether the item is urgent/high-priority (default: False)
                store_slugs: List of store slugs where the item can be purchased
                    (e.g., ["walmart", "target"])
                category: Optional category for organizing items (e.g., "dairy", "produce")

            Returns:
                A JSON string containing the details of the newly created item

            Example usage:
                create_item(
                    name="Organic Bananas",
                    slug="organic-bananas",
                    description="Bunch of organic bananas",
                    quantity=1,
                    unit="bunch",
                    price=2.99,
                    currency="USD",
                    urgent=False,
                    store_slugs=["grocery-store", "farmers-market"],
                    category="produce"
                )

            Example response structure:
                {
                  "success": {
                    "result": {
                      "item": {
                        "slug": "organic-bananas",
                        "name": "Organic Bananas",
                        "description": "Bunch of organic bananas",
                        "quantity": 1,
                        "unit": "bunch",
                        "price": 2.99,
                        "currency": "USD",
                        "urgent": false,
                        "purchased": false,
                        "purchase_date": null,
                        "purchase_history": [],
                        "notes": null,
                        "store_slugs": ["grocery-store", "farmers-market"],
                        "category": "produce",
                        "created_at": "2023-01-15T12:30:45Z",
                        "updated_at": "2023-01-15T12:30:45Z"
                      }
                    }
                  }
                }

            Error response when slug already exists:
                {
                  "error": {
                    "message": "Item with this slug already exists",
                    "code": "DUPLICATE_ENTITY",
                    "status": 409
                  }
                }

            Notes:
                - The slug must be unique across all items
                - The slug should contain only lowercase letters, numbers, and hyphens
                - At least name and slug are required; all other fields are optional
                - If store_slugs is provided, all stores must already exist in the system
            """
            await ctx.info(f"Creating item: {name} (slug: {slug})")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            kwargs = {
                "description": description,
                "quantity": quantity,
                "unit": unit,
                "price": price,
                "currency": currency,
                "urgent": urgent,
                "store_slugs": store_slugs,
                "category": category,
            }

            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            async with client.get_async_client() as async_client:
                response = await async_client.create_item_async(
                    name=name, slug=slug, **kwargs
                )
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def create_store(
            name: str,
            ctx: Context,
            slug: Optional[str] = None,
            address: Optional[str] = None,
            notes: Optional[str] = None,
            last_visit: Optional[str] = None,
        ) -> str:
            """
            Create a new store.

            This tool creates a new store entry in the shopping system. Stores are
            locations where shopping items can be purchased. Each store has a unique
            slug identifier that can be used to associate items with the store.

            Args:
                name: Store name (e.g., "Walmart", "Target", "Farmers Market")
                ctx: The MCP context object for logging and accessing the API client
                slug: Optional unique identifier for the store (if not provided, will be
                     generated from the name). Should be URL-friendly (lowercase, hyphens)
                address: Optional physical address or location of the store
                notes: Optional additional notes or information about the store
                last_visit: Optional date of last visit to the store (ISO format: YYYY-MM-DD)

            Returns:
                A JSON string containing the details of the newly created store

            Example usage:
                create_store(
                    name="Local Grocery",
                    slug="local-grocery",
                    address="123 Main St, Anytown, USA",
                    notes="Open 24 hours",
                    last_visit="2023-01-15"
                )

            Example response structure:
                {
                  "success": {
                    "result": {
                      "store": {
                        "slug": "local-grocery",
                        "name": "Local Grocery",
                        "address": "123 Main St, Anytown, USA",
                        "notes": "Open 24 hours",
                        "last_visit": "2023-01-15T00:00:00Z",
                        "created_at": "2023-01-20T14:30:45Z",
                        "updated_at": "2023-01-20T14:30:45Z"
                      }
                    }
                  }
                }

            Error response when slug already exists:
                {
                  "error": {
                    "message": "Store with this slug already exists",
                    "code": "DUPLICATE_ENTITY",
                    "status": 409
                  }
                }

            Notes:
                - If slug is not provided, the API will generate one based on the store name
                - The slug should contain only lowercase letters, numbers, and hyphens
                - The store name is the only required field
                - If last_visit is provided as a string, it should be in ISO format (YYYY-MM-DD)
            """
            await ctx.info(f"Creating store: {name}" + (f" (slug: {slug})" if slug else ""))

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            kwargs = {
                "slug": slug,
                "address": address,
                "notes": notes,
                "last_visit": last_visit,
            }

            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            async with client.get_async_client() as async_client:
                response = await async_client.create_store_async(
                    name=name, **kwargs
                )
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def mark_item_purchased(
            item_slug: str,
            ctx: Context,
            purchased: bool = True,
            store_slug: Optional[str] = None,
            price: Optional[float] = None,
            quantity: Optional[float] = None,
            notes: Optional[str] = None,
        ) -> str:
            """
            Mark a shopping item as purchased.

            This tool updates a shopping item's purchase status and can record additional
            purchase details such as price, quantity, and the store where it was purchased.
            The purchase information is added to the item's purchase history.

            Args:
                item_slug: The unique identifier for the shopping item to mark as purchased
                ctx: The MCP context object for logging and accessing the API client
                purchased: Whether the item is purchased (True) or not (False) (default: True)
                store_slug: Optional slug of the store where the item was purchased
                price: Optional price paid for the item
                quantity: Optional quantity purchased
                notes: Optional purchase notes or comments

            Returns:
                A JSON string containing the updated item details with purchase history

            Example usage:
                mark_item_purchased(
                    item_slug="milk",
                    purchased=True,
                    store_slug="grocery-store",
                    price=3.49,
                    quantity=1,
                    notes="On sale this week"
                )

            Example usage in batch operations:
                # When using with batch_operations, use "mark_purchased" as the operation type:
                batch_operations([
                    {
                        "type": "mark_purchased",  # Note: "mark_purchased", not "mark_item_purchased"
                        "params": {
                            "item_slug": "milk",
                            "price": 3.49,
                            "store_slug": "grocery-store"
                        }
                    }
                ])

            Example response structure:
                {
                  "success": {
                    "result": {
                      "item": {
                        "slug": "milk",
                        "name": "Milk",
                        "description": "Whole milk",
                        "quantity": 1,
                        "unit": "gallon",
                        "price": 3.99,
                        "currency": "USD",
                        "urgent": false,
                        "purchased": true,
                        "purchase_date": "2023-01-15T14:30:45Z",
                        "purchase_history": [
                          {
                            "date": "2023-01-15T14:30:45Z",
                            "price": 3.49,
                            "quantity": 1,
                            "store_slug": "grocery-store",
                            "notes": "On sale this week"
                          }
                        ],
                        "notes": null,
                        "store_slugs": ["grocery-store"],
                        "category": "dairy",
                        "created_at": "2023-01-10T12:30:45Z",
                        "updated_at": "2023-01-15T14:30:45Z"
                      }
                    }
                  }
                }

            Error response when item not found:
                {
                  "error": {
                    "message": "Item not found",
                    "code": "NOT_FOUND",
                    "status": 404
                  }
                }

            Notes:
                - Setting purchased=False will clear the purchase status of the item
                - If store_slug is provided, the store must already exist in the system
                - The purchase date is automatically set to the current time
                - Purchase history is maintained even when an item is marked as not purchased
                - When using this function within batch_operations, use "mark_purchased"
                  (not "mark_item_purchased") as the operation type
            """
            purchase_status = "purchased" if purchased else "not purchased"
            await ctx.info(f"Marking item '{item_slug}' as {purchase_status}")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            kwargs = {
                "purchased": purchased,
                "store_slug": store_slug,
                "price": price,
                "quantity": quantity,
                "notes": notes,
            }

            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            async with client.get_async_client() as async_client:
                response = await async_client.mark_item_purchased_async(
                    item_slug=item_slug, **kwargs
                )
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def batch_operations(
            operations: List[Dict[str, Any]], ctx: Context
        ) -> str:
            """
            Execute multiple operations in parallel.

            This tool allows running multiple API operations concurrently in a single request,
            improving efficiency for bulk operations. Each operation is executed asynchronously,
            and the results are collected and returned together.

            Args:
                operations: List of operations to execute. Each operation is a dictionary with:
                    - type: Operation type as a string. Supported types:
                        - "create_item": Create a new shopping item
                        - "update_item": Update an existing shopping item
                        - "mark_purchased": Mark an item as purchased (same as mark_item_purchased tool)
                        - "create_store": Create a new store
                        - "create_task": Create a new task
                        - "delete_task": Delete an existing task
                    - params: Dictionary of parameters for the operation (corresponding to the
                      parameters of the respective API method)
                ctx: The MCP context object for logging and accessing the API client

            Returns:
                A JSON string containing the results of all operations, with success/failure
                status for each operation

            Example usage:
                batch_operations([
                    {
                        "type": "create_store",
                        "params": {
                            "name": "Local Grocery",
                            "slug": "local-grocery",
                            "address": "123 Main St, Anytown, USA"
                        }
                    },
                    {
                        "type": "create_item",
                        "params": {
                            "name": "Apples",
                            "slug": "apples",
                            "description": "Red apples",
                            "quantity": 5,
                            "unit": "pieces",
                            "price": 2.99,
                            "currency": "USD",
                            "store_slugs": ["local-grocery"]
                        }
                    },
                    {
                        "type": "create_item",
                        "params": {
                            "name": "Oranges",
                            "slug": "oranges",
                            "quantity": 3,
                            "unit": "pieces",
                            "store_slugs": ["local-grocery"]
                        }
                    },
                    {
                        "type": "mark_purchased",
                        "params": {
                            "item_slug": "milk",
                            "price": 3.50,
                            "store_slug": "local-grocery"
                        }
                    }
                ])

            Example response structure:
                [
                  {
                    "success": true,
                    "result": {
                      "success": {
                        "result": {
                          "item": {
                            "slug": "apples",
                            "name": "Apples",
                            ...
                          }
                        }
                      }
                    },
                    "operation": {
                      "type": "create_item",
                      "params": {...}
                    }
                  },
                  {
                    "success": true,
                    "result": {
                      "success": {
                        "result": {
                          "item": {
                            "slug": "oranges",
                            "name": "Oranges",
                            ...
                          }
                        }
                      }
                    },
                    "operation": {
                      "type": "create_item",
                      "params": {...}
                    }
                  },
                  {
                    "success": true,
                    "result": {
                      "success": {
                        "result": {
                          "item": {
                            "slug": "milk",
                            "purchased": true,
                            ...
                          }
                        }
                      }
                    },
                    "operation": {
                      "type": "mark_purchased",
                      "params": {...}
                    }
                  }
                ]

            Error handling:
                If any operation fails, its entry in the results will have "success": false
                and an error message:

                {
                  "success": false,
                  "error": "Item with slug 'apples' already exists",
                  "operation": {
                    "type": "create_item",
                    "params": {...}
                  }
                }

            Notes:
                - All operations are executed in parallel for better performance
                - The order of results matches the order of operations in the input
                - Each operation must have the correct parameters for its type
                - Progress is reported during execution via the context object
                - If an operation is invalid or missing required parameters, it will
                  be included in the results with an error message
            """
            await ctx.info(f"Executing batch operations: {len(operations)} operations")

            if not operations:
                return json.dumps({"error": "No operations provided"})

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                # Create tasks for all operations
                tasks = []

                for i, op in enumerate(operations):
                    op_type = op.get("type")
                    params = op.get("params", {})

                    await ctx.report_progress(i, len(operations))

                    if op_type == "create_item":
                        task = async_client.create_item_async(**params)
                    elif op_type == "update_item":
                        item_slug = params.pop("item_slug", None)
                        if item_slug:
                            task = async_client.update_item_async(item_slug, **params)
                        else:
                            tasks.append(
                                {"error": "Missing item_slug for update_item operation"}
                            )
                            continue
                    elif op_type == "mark_purchased":
                        item_slug = params.pop("item_slug", None)
                        if item_slug:
                            task = async_client.mark_item_purchased_async(
                                item_slug, **params
                            )
                        else:
                            tasks.append(
                                {
                                    "error": "Missing item_slug for mark_purchased operation"
                                }
                            )
                            continue
                    elif op_type == "create_store":
                        task = async_client.create_store_async(**params)
                    elif op_type == "create_task":
                        task = async_client.create_task_async(**params)
                    elif op_type == "delete_task":
                        task_slug = params.get("task_slug")
                        if task_slug:
                            task = async_client.delete_task_async(task_slug)
                        else:
                            tasks.append(
                                {"error": "Missing task_slug for delete_task operation"}
                            )
                            continue
                    else:
                        tasks.append({"error": f"Unknown operation type: {op_type}"})
                        continue

                    tasks.append(task)

                # Execute all tasks concurrently
                await ctx.info(f"Executing {len(tasks)} operations concurrently")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append(
                            {
                                "success": False,
                                "error": str(result),
                                "operation": operations[i]
                                if i < len(operations)
                                else "unknown",
                            }
                        )
                    else:
                        processed_results.append(
                            {
                                "success": True,
                                "result": result,
                                "operation": operations[i]
                                if i < len(operations)
                                else "unknown",
                            }
                        )

                return json.dumps(processed_results, indent=2)

    def run(self, transport: Literal["stdio", "sse", "streamable-http"]):
        """
        Run the MCP server with the specified transport method.

        This method starts the MCP server and begins accepting requests using
        the specified transport method. The server will continue running until
        interrupted (e.g., by a keyboard interrupt or process termination).

        Args:
            transport: The transport method to use for receiving requests and
                      sending responses. Options:
                      - "stdio": Standard input/output, useful for CLI tools, testing,
                                 or direct integration with language models
                      - "sse": Server-Sent Events, for web-based interfaces with
                               real-time updates
                      - "streamable-http": HTTP streaming for more complex integrations
                                          with advanced web applications

        Returns:
            None. This method runs indefinitely until interrupted.

        Example:
            # Run with standard I/O transport (for CLI or testing)
            server.run("stdio")

            # Run with SSE transport (for web applications)
            server.run("sse")
        """
        self.mcp.run(transport=transport)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Async MCP Server for Tasks & Shopping API"
    )
    parser.add_argument("--api-url", required=True, help="Base URL for the API")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type (stdio or SSE)",
    )

    args = parser.parse_args()

    server = AsyncMCPTasksShoppingServer(
        api_url=args.api_url, api_key=args.api_key, port=args.port
    )

    # Set additional options if using SSE transport

    # Run the server
    server.run(args.transport)
