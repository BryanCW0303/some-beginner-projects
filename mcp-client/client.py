import asyncio
import os
import json
from typing import Optional
from contextlib import AsyncExitStack

from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load .env file to keep API key secure
load_dotenv()

class MCPClient:
    def __init__(self):
        """Initialize MCP Client"""
        self.exit_stack = AsyncExitStack()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # Read DeepSeek API Key
        self.base_url = os.getenv("BASE_URL")  # Read BASE URL
        self.model = os.getenv("MODEL")  # Read model name
        if not self.deepseek_api_key:
            raise ValueError("❌ DeepSeek API Key not found, please set DEEPSEEK_API_KEY in .env file")
        self.client = OpenAI(api_key=self.deepseek_api_key, base_url=self.base_url)
        # Create OpenAI (DeepSeek-compatible) client
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server and list available tools"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # Start MCP server and establish communication
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools from the MCP server
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server, available tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        Use LLM to process query and call available MCP tools (Function Calling)
        """
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools,
        )

        # Handle returned response
        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Execute the tool
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"\n[Calling tool] {tool_name} with args {tool_args}\n")

            # Append model request and tool response to messages
            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id,
            })

            # Send updated messages back to the model to generate final response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content

        return content.message.content

    async def chat_loop(self):
        """Run interactive chat loop"""
        print("\n🤖 MCP client started! Type `quit` to exit")
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)  # Send query to DeepSeek API
                print(f"\n🤖 DeepSeek: {response}")
            except Exception as e:
                print(f"\n⚠️ Error occurred: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
