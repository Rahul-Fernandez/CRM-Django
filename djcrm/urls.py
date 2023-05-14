from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.views import (
    LoginView, 
    LogoutView, 
    PasswordResetView, 
    PasswordResetDoneView,
    PasswordResetConfirmView,
    PasswordResetCompleteView
)
from django.urls import path, include
from leads.views import landing_page, LandingPageView, SignupView, DashboardView
from django.views.generic import TemplateView
from Clustering.views import *
urlpatterns = [
    path('',TemplateView.as_view(template_name="landing.html"), name = 'landing'),
    path('admin/', admin.site.urls),
#   path('', LandingPageView.as_view(), name='landing-page'),
    # path('',TemplateView.as_view(template_name="landing.html"), name = 'landing'),
    # path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('dashboard/',TemplateView.as_view(template_name="statdash.html"), name = 'dashboard'),
    path('aitools/',TemplateView.as_view(template_name="aitools.html"), name = 'aitools'),
    # path('aitools/',  include('aitools.urls', namespace="aitools")),
    path('leads/',  include('leads.urls', namespace="leads")),
    path('agents/',  include('agents.urls', namespace="agents")),
    path('signup/', SignupView.as_view(), name='signup'),
    path('reset-password/', PasswordResetView.as_view(), name='reset-password'),
    path('password-reset-done/', PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password-reset-complete/', PasswordResetCompleteView.as_view(), name='password_reset_complete'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    
    path('preprocessing/', preprocessing, name='preprocessing'),
    path('checker_page/', checker_page, name='checker_page'),
    path('chooseMethod/', chooseMethod, name='chooseMethod'),
    path('classification/', classification, name='classification'),
    path('clustering/', clustering, name='clustering'),
     
     path('forecast/',TemplateView.as_view(template_name="forecast.html"), name = 'forecast')


]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

