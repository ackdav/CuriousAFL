/**
 * Autogenerated by Thrift Compiler (0.12.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#ifndef RND_SERVICE_H
#define RND_SERVICE_H

#include <thrift/c_glib/processor/thrift_dispatch_processor.h>

#include "rnd_types.h"

/* RndService service interface */
typedef struct _RndServiceIf RndServiceIf;  /* dummy object */

struct _RndServiceIfInterface
{
  GTypeInterface parent;

  gboolean (*init_model) (RndServiceIf *iface, gdouble* _return, GError **error);
  gboolean (*veto) (RndServiceIf *iface, gdouble* _return, const gchar * seed, const gchar * mode, GError **error);
};
typedef struct _RndServiceIfInterface RndServiceIfInterface;

GType rnd_service_if_get_type (void);
#define TYPE_RND_SERVICE_IF (rnd_service_if_get_type())
#define RND_SERVICE_IF(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), TYPE_RND_SERVICE_IF, RndServiceIf))
#define IS_RND_SERVICE_IF(obj) (G_TYPE_CHECK_INSTANCE_TYPE ((obj), TYPE_RND_SERVICE_IF))
#define RND_SERVICE_IF_GET_INTERFACE(inst) (G_TYPE_INSTANCE_GET_INTERFACE ((inst), TYPE_RND_SERVICE_IF, RndServiceIfInterface))

gboolean rnd_service_if_init_model (RndServiceIf *iface, gdouble* _return, GError **error);
gboolean rnd_service_if_veto (RndServiceIf *iface, gdouble* _return, const gchar * seed, const gchar * mode, GError **error);

/* RndService service client */
struct _RndServiceClient
{
  GObject parent;

  ThriftProtocol *input_protocol;
  ThriftProtocol *output_protocol;
};
typedef struct _RndServiceClient RndServiceClient;

struct _RndServiceClientClass
{
  GObjectClass parent;
};
typedef struct _RndServiceClientClass RndServiceClientClass;

GType rnd_service_client_get_type (void);
#define TYPE_RND_SERVICE_CLIENT (rnd_service_client_get_type())
#define RND_SERVICE_CLIENT(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), TYPE_RND_SERVICE_CLIENT, RndServiceClient))
#define RND_SERVICE_CLIENT_CLASS(c) (G_TYPE_CHECK_CLASS_CAST ((c), TYPE_RND_SERVICE_CLIENT, RndServiceClientClass))
#define RND_SERVICE_IS_CLIENT(obj) (G_TYPE_CHECK_INSTANCE_TYPE ((obj), TYPE_RND_SERVICE_CLIENT))
#define RND_SERVICE_IS_CLIENT_CLASS(c) (G_TYPE_CHECK_CLASS_TYPE ((c), TYPE_RND_SERVICE_CLIENT))
#define RND_SERVICE_CLIENT_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), TYPE_RND_SERVICE_CLIENT, RndServiceClientClass))

gboolean rnd_service_client_init_model (RndServiceIf * iface, gdouble* _return, GError ** error);
gboolean rnd_service_client_send_init_model (RndServiceIf * iface, GError ** error);
gboolean rnd_service_client_recv_init_model (RndServiceIf * iface, gdouble* _return, GError ** error);
gboolean rnd_service_client_veto (RndServiceIf * iface, gdouble* _return, const gchar * seed, const gchar * mode, GError ** error);
gboolean rnd_service_client_send_veto (RndServiceIf * iface, const gchar * seed, const gchar * mode, GError ** error);
gboolean rnd_service_client_recv_veto (RndServiceIf * iface, gdouble* _return, GError ** error);
void rnd_service_client_set_property (GObject *object, guint property_id, const GValue *value, GParamSpec *pspec);
void rnd_service_client_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec);

/* RndService handler (abstract base class) */
struct _RndServiceHandler
{
  GObject parent;
};
typedef struct _RndServiceHandler RndServiceHandler;

struct _RndServiceHandlerClass
{
  GObjectClass parent;

  gboolean (*init_model) (RndServiceIf *iface, gdouble* _return, GError **error);
  gboolean (*veto) (RndServiceIf *iface, gdouble* _return, const gchar * seed, const gchar * mode, GError **error);
};
typedef struct _RndServiceHandlerClass RndServiceHandlerClass;

GType rnd_service_handler_get_type (void);
#define TYPE_RND_SERVICE_HANDLER (rnd_service_handler_get_type())
#define RND_SERVICE_HANDLER(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), TYPE_RND_SERVICE_HANDLER, RndServiceHandler))
#define IS_RND_SERVICE_HANDLER(obj) (G_TYPE_CHECK_INSTANCE_TYPE ((obj), TYPE_RND_SERVICE_HANDLER))
#define RND_SERVICE_HANDLER_CLASS(c) (G_TYPE_CHECK_CLASS_CAST ((c), TYPE_RND_SERVICE_HANDLER, RndServiceHandlerClass))
#define IS_RND_SERVICE_HANDLER_CLASS(c) (G_TYPE_CHECK_CLASS_TYPE ((c), TYPE_RND_SERVICE_HANDLER))
#define RND_SERVICE_HANDLER_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), TYPE_RND_SERVICE_HANDLER, RndServiceHandlerClass))

gboolean rnd_service_handler_init_model (RndServiceIf *iface, gdouble* _return, GError **error);
gboolean rnd_service_handler_veto (RndServiceIf *iface, gdouble* _return, const gchar * seed, const gchar * mode, GError **error);

/* RndService processor */
struct _RndServiceProcessor
{
  ThriftDispatchProcessor parent;

  /* protected */
  RndServiceHandler *handler;
  GHashTable *process_map;
};
typedef struct _RndServiceProcessor RndServiceProcessor;

struct _RndServiceProcessorClass
{
  ThriftDispatchProcessorClass parent;

  /* protected */
  gboolean (*dispatch_call) (ThriftDispatchProcessor *processor,
                             ThriftProtocol *in,
                             ThriftProtocol *out,
                             gchar *fname,
                             gint32 seqid,
                             GError **error);
};
typedef struct _RndServiceProcessorClass RndServiceProcessorClass;

GType rnd_service_processor_get_type (void);
#define TYPE_RND_SERVICE_PROCESSOR (rnd_service_processor_get_type())
#define RND_SERVICE_PROCESSOR(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), TYPE_RND_SERVICE_PROCESSOR, RndServiceProcessor))
#define IS_RND_SERVICE_PROCESSOR(obj) (G_TYPE_CHECK_INSTANCE_TYPE ((obj), TYPE_RND_SERVICE_PROCESSOR))
#define RND_SERVICE_PROCESSOR_CLASS(c) (G_TYPE_CHECK_CLASS_CAST ((c), TYPE_RND_SERVICE_PROCESSOR, RndServiceProcessorClass))
#define IS_RND_SERVICE_PROCESSOR_CLASS(c) (G_TYPE_CHECK_CLASS_TYPE ((c), TYPE_RND_SERVICE_PROCESSOR))
#define RND_SERVICE_PROCESSOR_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), TYPE_RND_SERVICE_PROCESSOR, RndServiceProcessorClass))

#endif /* RND_SERVICE_H */
