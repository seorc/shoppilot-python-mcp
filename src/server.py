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
    """Context for API client access"""

    sync_client: APIClient
    async_client: Optional[AsyncAPIClient] = None


@asynccontextmanager
async def api_lifespan(server: FastMCP) -> AsyncIterator[ApiContext]:
    """
    Manage API client lifecycle with type-safe context

    This lifecycle handler initializes and properly disposes of the API client,
    ensuring connections are properly managed.
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
    def __init__(self, api_url: str, api_key: Optional[str] = None, port=8000):
        """
        Initialize the MCP server for the Tasks & Shopping API with async support

        Args:
            api_url: Base URL for the API
            api_key: Optional API key for authentication
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
        """Register all resources for the MCP server"""

        @self.mcp.tool()
        async def get_store_items(
            store_slug: str,
            page: int,
            ctx: Context,
            limit: Optional[int] = None,
            purchased: Optional[bool] = None,
            urgent: Optional[bool] = None,
        ) -> str:
            """Get shopping items for a specific store"""
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
            """Get all urgent shopping items"""
            await ctx.info("Fetching urgent items")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                response = await async_client.get_urgent_items_async()
                return json.dumps(response, indent=2)

        @self.mcp.tool()
        async def get_item(item_slug: str, ctx: Context) -> str:
            """Get a single shopping item by slug"""
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
            """Search for shopping items by name or text match"""
            await ctx.info(f"Searching for items: {query} (limit: {limit})")

            api_context = ctx.request_context.lifespan_context
            client = api_context.sync_client

            async with client.get_async_client() as async_client:
                response = await async_client.search_items_async(
                    query=query, limit=limit
                )
                return json.dumps(response, indent=2)

    def register_tools(self):
        """Register all tools for the MCP server"""

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
            Create a new shopping item

            Args:
                name: Item name
                slug: Item slug (unique identifier)
                description: Optional item description
                quantity: Optional quantity
                unit: Optional unit (e.g., "kg", "pieces")
                price: Optional price
                currency: Currency code (default: "MXN")
                urgent: Whether the item is urgent (default: False)
                store_slugs: List of store slugs where the item is available
                category: Optional category

            Returns:
                JSON response with the created item details
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
        async def batch_operations(
            operations: List[Dict[str, Any]], ctx: Context
        ) -> str:
            """
            Execute multiple operations in parallel

            Args:
                operations: List of operations to execute
                    Each operation should have:
                    - type: Operation type (e.g., "create_item", "mark_purchased")
                    - params: Parameters for the operation

            Returns:
                JSON response with the results of all operations
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
        """Run the MCP server"""
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
