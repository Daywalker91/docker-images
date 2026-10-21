# Base: official php:8.5-apache from Docker Hub
# Feste Minor-Version: die täglichen Builds holen Sicherheits-Updates (8.5.x),
# aber kein unbemerktes PHP 8.6/9 – ein Major-Wechsel wird bewusst hier umgestellt.
FROM php:8.5-apache

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="Custom PHP/Apache image"
LABEL org.opencontainers.image.licenses="MIT"

RUN mv "$PHP_INI_DIR/php.ini-production" "$PHP_INI_DIR/php.ini"

# Apache-Module, die die Anwendungen per .htaccess nutzen:
#   rewrite – "RewriteRule" (Routing, Sperren einzelner Pfade)
#   headers – "Header set" (Sicherheits-Header auch für statische Dateien)
RUN a2enmod rewrite headers

RUN pecl install redis \
 && docker-php-ext-install mysqli pdo pdo_mysql \
 && docker-php-ext-enable mysqli pdo pdo_mysql redis
