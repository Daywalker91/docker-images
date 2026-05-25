# Base: official php:apache from Docker Hub
FROM php:apache

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="Custom PHP/Apache image"
LABEL org.opencontainers.image.licenses="MIT"

RUN mv "$PHP_INI_DIR/php.ini-production" "$PHP_INI_DIR/php.ini"

RUN docker-php-ext-install mysqli pdo pdo_mysql \
 && docker-php-ext-enable mysqli pdo pdo_mysql
