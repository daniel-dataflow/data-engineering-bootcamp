from django.http import HttpResponse

def index(request):
    # return HttpResponse("<h1>Hello, world</h1>")
    return HttpResponse("""
        <h1>Hello, world!</h1>
        <a href="/hello01/">hello01</a>
    """)