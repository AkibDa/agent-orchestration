import asyncio
from location.resolver import extract_location
async def main():
    res = extract_location("fishing area")
    print(res)
if __name__ == "__main__":
    asyncio.run(main())
