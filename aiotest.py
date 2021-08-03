
def get_whatiwant_from_html(html_content):
    pass

async def get_data(session, url, params=None):
    loop = asyncio.get_event_loop()
    async with session.get(url, headers=HEADERS, params=params) as response:
        html = await response.text()
        data = await loop.run_in_executor(None, partial(get_whatiwant_from_html, html))
        return data

async def get_data_from_urls(urls):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for url in urls:
            tasks.append(get_data(session, url))
        result_data = await asyncio.gather(*tasks)
    return result_data

executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)
asyncio_loop.set_default_executor(executor)
results = asyncio_loop.run_until_complete(get_data_from_urls(urls))